"""
直接测试推理功能（不使用 API 封装）
Directly test the inference functionality (without the API wrapper)
"""

import os
import sys
import time

def test_direct_inference():
    """直接使用 infer.py 进行测试
    Test directly using infer.py
    """

    print("=" * 60)
    # original text:"DiffV2IR 直接推理测试"
    print("DiffV2IR direct inference test")
    print("=" * 60)

    # 准备测试参数
    # Prepare the test parameters
    input_dir = "data_for_diffv2ir/input"
    output_dir = "test_output_direct"
    checkpoint = "pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt"

    # 创建输出目录
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # 构建命令
    # Build the command
    cmd = f"""python infer.py \
        --input {input_dir} \
        --output {output_dir} \
        --ckpt {checkpoint} \
        --steps 10 \
        --resolution 256 \
        --seed 42
    """
    
    # original text:"\n执行命令:\n{cmd}"
    print(f"\nRunning command:\n{cmd}")
    # original text:"\n开始推理测试（使用少量步数）..."
    print("\nStarting the inference test (using a small number of steps)...")

    start_time = time.time()

    # 执行推理
    # Run the inference
    exit_code = os.system(cmd)

    elapsed_time = time.time() - start_time

    if exit_code == 0:
        # original text:"\n✓ 推理成功！耗时: {elapsed_time:.2f} 秒"
        print(f"\n✓ Inference succeeded! Elapsed time: {elapsed_time:.2f} s")

        # 检查输出文件
        # Check the output files
        output_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
        if output_files:
            # original text:"✓ 生成了 {len(output_files)} 个输出文件:"
            print(f"✓ Generated {len(output_files)} output files:")
            for f in output_files[:5]:  # 只显示前5个 (only show the first 5)
                size = os.path.getsize(os.path.join(output_dir, f)) / 1024
                print(f"  - {f} ({size:.2f} KB)")

        return True
    else:
        # original text:"\n✗ 推理失败，退出码: {exit_code}"
        print(f"\n✗ Inference failed, exit code: {exit_code}")
        return False

if __name__ == "__main__":
    success = test_direct_inference()

    print("\n" + "=" * 60)
    if success:
        # original text:"测试通过！DiffV2IR 基本功能正常"
        print("Test passed! DiffV2IR basic functionality is working")
        # original text:"\n使用方式:"
        print("\nUsage:")
        # original text:"1. 直接使用 infer.py 脚本"
        print("1. Use the infer.py script directly")
        # original text:"2. API 封装需要解决依赖版本冲突问题"
        print("2. The API wrapper requires resolving dependency version conflicts")
    else:
        # original text:"测试失败，请检查错误信息"
        print("Test failed, please check the error messages")