"""
DiffV2IR API - 封装的可见光转红外推理接口
Encapsulated visible-light to infrared (RGB->TIR) inference interface
提供简单的函数调用方式进行图像转换
Provides a simple function-call interface for image conversion
"""

import os
import sys
import math
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from omegaconf import OmegaConf
from einops import rearrange
from torch import autocast
import k_diffusion as K
import einops
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import gc
import tempfile
from typing import Optional, Dict, Any

sys.path.append("./stable_diffusion")
from stable_diffusion.ldm.util import instantiate_from_config
from blip_models.blip import blip_decoder


class CFGDenoiser(nn.Module):
    """配置引导去噪器
    Classifier-free-guidance (CFG) denoiser
    """
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0],
                                       uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], 
                                    uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0],
                                    cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + \
               image_cfg_scale * (out_img_cond - out_seg_cond) + \
               seg_cfg_scale * (out_seg_cond - out_uncond)


class DiffV2IR:
    """DiffV2IR 模型封装类
    Wrapper class for the DiffV2IR model
    """
    
    def __init__(self, 
                 config_path: str = "configs/generate.yaml",
                 checkpoint_path: str = "pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
                 device: str = "cuda",
                 use_fp16: bool = False,
                 load_blip: bool = True):
        """
        初始化 DiffV2IR 模型
        Initialize the DiffV2IR model

        Args:
            config_path: 模型配置文件路径
                Path to the model configuration file
            checkpoint_path: 模型权重文件路径
                Path to the model weight (checkpoint) file
            device: 计算设备 (cuda/cpu)
                Compute device (cuda/cpu)
            use_fp16: 是否使用半精度推理
                Whether to use half-precision (FP16) inference
            load_blip: 是否加载 BLIP 模型用于图像描述
                Whether to load the BLIP model for image captioning
        """
        self.device = device
        self.use_fp16 = use_fp16
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # 加载模型
        # Load the model
        self._load_model()

        # 加载 BLIP
        # Load BLIP
        self.blip_model = None
        if load_blip:
            self._load_blip()
    
    def _load_model(self):
        """加载 DiffV2IR 模型
        Load the DiffV2IR model
        """
        print(f"Loading DiffV2IR model from {self.checkpoint_path}")

        # 加载配置
        # Load the configuration
        config = OmegaConf.load(self.config_path)

        # 加载模型
        # Load the model
        pl_sd = torch.load(self.checkpoint_path, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(sd, strict=False)
        
        # 转换精度和设备
        # Convert precision and move to device
        if self.use_fp16:
            self.model = self.model.half()
            print("Using FP16 (half precision) mode")

        self.model = self.model.to(self.device)
        self.model.eval()

        # 准备包装器
        # Prepare the wrapper
        model_wrap = K.external.CompVisDenoiser(self.model)
        self.model_wrap_cfg = CFGDenoiser(model_wrap)
        self.null_token = self.model.get_learned_conditioning([""])

        # 清理内存
        # Clean up memory
        del pl_sd, sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _load_blip(self):
        """加载 BLIP 模型
        Load the BLIP model
        """
        print("Loading BLIP model for image captioning...")
        self.blip_model = blip_decoder(
            pretrained="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth",
            image_size=384, 
            vit='base'
        )
        self.blip_model.eval()
        if self.use_fp16:
            self.blip_model = self.blip_model.half()
        self.blip_model = self.blip_model.to(self.device)
    
    def convert(self,
                input_path: str,
                output_path: Optional[str] = None,
                mask_path: Optional[str] = None,
                resolution: int = 512,
                steps: int = 50,
                cfg_text: float = 7.5,
                cfg_image: float = 1.5,
                cfg_seg: float = 1.5,
                seed: Optional[int] = None,
                edit_prompt: Optional[str] = None) -> str:
        """
        将可见光图像转换为红外图像
        Convert a visible-light (RGB) image to an infrared (TIR) image

        Args:
            input_path: 输入图像路径
                Path to the input image
            output_path: 输出图像路径（可选，如果不提供则自动生成）
                Output image path (optional; auto-generated if not provided)
            mask_path: 分割图路径（可选）
                Path to the segmentation map (optional)
            resolution: 处理分辨率
                Processing resolution
            steps: 去噪步数
                Number of denoising steps
            cfg_text: 文本引导强度
                Text guidance strength
            cfg_image: 图像条件强度
                Image-conditioning strength
            cfg_seg: 分割图引导强度
                Segmentation-map guidance strength
            seed: 随机种子
                Random seed
            edit_prompt: 自定义编辑提示词（可选）
                Custom edit prompt (optional)

        Returns:
            str: 输出图像路径
                Path to the output image
        """
        # 检查输入文件
        # Check the input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # 生成输出路径
        # Generate the output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = os.path.dirname(input_path)
            output_path = os.path.join(output_dir, f"{base_name}_infrared.png")

        # 确保输出目录存在
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # 生成或使用提供的编辑提示词
        # Generate or use the provided edit prompt
        if edit_prompt is None:
            if self.blip_model is not None:
                # 使用 BLIP 生成描述
                # Use BLIP to generate a description
                image = self._load_image_for_blip(input_path)
                with torch.no_grad():
                    if self.use_fp16:
                        with autocast("cuda", dtype=torch.float16):
                            caption = self.blip_model.generate(image, sample=True, 
                                                              top_p=0.9, max_length=20, min_length=5)
                    else:
                        caption = self.blip_model.generate(image, sample=True, 
                                                          top_p=0.9, max_length=20, min_length=5)
                edit_prompt = f"turn the visible image of {caption[0]} into infrared"
            else:
                edit_prompt = "turn the RGB image into the infrared one"
        
        print(f"Edit prompt: {edit_prompt}")
        
        # 加载和预处理图像
        # Load and preprocess the image
        input_image = Image.open(input_path).convert("RGB")

        # 加载或创建分割图
        # Load or create the segmentation map
        if mask_path and os.path.exists(mask_path):
            input_seg = Image.open(mask_path).convert("RGB")
        else:
            # 如果没有分割图，使用原图
            # If there is no segmentation map, use the original image
            input_seg = input_image.copy()
        
        # 调整图像尺寸
        # Adjust the image size
        width, height = input_image.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        
        input_image = ImageOps.fit(input_image, (width, height), 
                                   method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        input_seg = ImageOps.fit(input_seg, (width, height), 
                                method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        
        # 执行推理
        # Run inference
        with torch.no_grad():
            if self.use_fp16:
                context = autocast("cuda", dtype=torch.float16)
            else:
                context = autocast("cuda")

            with context, self.model.ema_scope():
                # 准备条件
                # Prepare the conditioning
                cond = {}
                cond["c_crossattn"] = [self.model.get_learned_conditioning([edit_prompt])]

                # 预处理图像
                # Preprocess the image
                input_image_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_seg_tensor = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                
                if self.use_fp16:
                    input_image_tensor = input_image_tensor.half()
                    input_seg_tensor = input_seg_tensor.half()
                
                input_image_tensor = rearrange(input_image_tensor, "h w c -> 1 c h w").to(self.device)
                input_seg_tensor = rearrange(input_seg_tensor, "h w c -> 1 c h w").to(self.device)
                
                cond["c_concat1"] = [self.model.encode_first_stage(input_image_tensor).mode()]
                cond["c_concat2"] = [self.model.encode_first_stage(input_seg_tensor).mode()]
                
                # 准备无条件输入
                # Prepare the unconditional input
                uncond = {}
                uncond["c_crossattn"] = [self.null_token]
                uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]

                # 采样
                # Sampling: get the sigma schedule
                sigmas = self.model_wrap_cfg.inner_model.get_sigmas(steps)
                
                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": cfg_text,
                    "image_cfg_scale": cfg_image,
                    "seg_cfg_scale": cfg_seg,
                }
                
                # 设置随机种子
                # Set the random seed
                if seed is None:
                    seed = random.randint(0, 100000)
                torch.manual_seed(seed)

                # 生成初始噪声
                # Generate the initial noise
                z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]

                # 采样
                # Sample (denoise) with Euler-ancestral
                z = K.sampling.sample_euler_ancestral(self.model_wrap_cfg, z, sigmas, extra_args=extra_args)

                # 解码
                # Decode the latent back to image space
                x = self.model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
        
        # 保存结果
        # Save the result
        edited_image.save(output_path)
        print(f"Result saved to: {output_path}")

        # 清理显存
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_path

    def _load_image_for_blip(self, image_path: str):
        """为 BLIP 加载和预处理图像
        Load and preprocess an image for BLIP
        """
        raw_image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        image = transform(raw_image).unsqueeze(0)
        if self.use_fp16:
            image = image.half()
        return image.to(self.device)
    
    def batch_convert(self,
                     input_folder: str,
                     output_folder: str,
                     mask_folder: Optional[str] = None,
                     **kwargs) -> Dict[str, str]:
        """
        批量转换图像
        Batch-convert a folder of images

        Args:
            input_folder: 输入图像文件夹
                Input image folder
            output_folder: 输出图像文件夹
                Output image folder
            mask_folder: 分割图文件夹（可选）
                Segmentation-map folder (optional)
            **kwargs: 其他参数传递给 convert 方法
                Other parameters passed through to the convert method

        Returns:
            Dict[str, str]: 输入文件名到输出路径的映射
                Mapping from input filename to output path
        """
        os.makedirs(output_folder, exist_ok=True)

        # 获取所有图像文件
        # Collect all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        results = {}
        
        for filename in os.listdir(input_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                mask_path = None
                if mask_folder:
                    base_name = os.path.splitext(filename)[0]
                    mask_path = os.path.join(mask_folder, f"{base_name}.png")
                    if not os.path.exists(mask_path):
                        mask_path = None
                
                try:
                    print(f"\nProcessing: {filename}")
                    output = self.convert(input_path, output_path, mask_path, **kwargs)
                    results[filename] = output
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results[filename] = None
        
        return results


# 简单的函数接口
# Simple function interface
def diffv2ir_convert(input_path: str,
                     output_path: Optional[str] = None,
                     config_path: str = "configs/generate.yaml",
                     checkpoint_path: str = "pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
                     resolution: int = 512,
                     steps: int = 50,
                     use_fp16: bool = False,
                     **kwargs) -> str:
    """
    便捷的函数接口，用于单张图像转换
    Convenience function interface for converting a single image

    Args:
        input_path: 输入图像路径
            Path to the input image
        output_path: 输出图像路径（可选）
            Output image path (optional)
        config_path: 模型配置文件路径
            Path to the model configuration file
        checkpoint_path: 模型权重路径
            Path to the model weights
        resolution: 处理分辨率
            Processing resolution
        steps: 去噪步数
            Number of denoising steps
        use_fp16: 是否使用半精度
            Whether to use half precision
        **kwargs: 其他参数
            Other parameters

    Returns:
        str: 输出图像路径
            Path to the output image
    """
    # 创建模型实例
    # Create a model instance
    model = DiffV2IR(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        use_fp16=use_fp16,
        load_blip=True
    )
    
    # 执行转换
    # Run the conversion
    output = model.convert(
        input_path=input_path,
        output_path=output_path,
        resolution=resolution,
        steps=steps,
        **kwargs
    )

    # 清理模型
    # Clean up the model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return output


if __name__ == "__main__":
    # 测试代码
    # Test code
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Output image path")
    parser.add_argument("--config", default="configs/generate.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt", 
                       help="Checkpoint path")
    parser.add_argument("--resolution", type=int, default=512, help="Processing resolution")
    parser.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mode")
    
    args = parser.parse_args()
    
    # 使用简单函数接口
    # Use the simple function interface
    output_path = diffv2ir_convert(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        resolution=args.resolution,
        steps=args.steps,
        use_fp16=args.fp16
    )
    
    print(f"Conversion complete! Output saved to: {output_path}")