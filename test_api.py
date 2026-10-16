"""
测试 DiffV2IR API 功能
Test the DiffV2IR API functionality
"""

import os
import sys
import time
from diffv2ir_api import diffv2ir_convert, DiffV2IR

def test_simple_api():
    """测试简单函数接口
    Test the simple function interface
    """
    print("=" * 60)
    # original text:"测试简单函数接口 diffv2ir_convert"
    print("Testing the simple function interface diffv2ir_convert")
    print("=" * 60)

    # 测试单张图像转换
    # Test single-image conversion
    input_path = "data_for_diffv2ir/input/DJI_0061.png"

    if not os.path.exists(input_path):
        # original text:"错误：测试图像不存在 {input_path}"
        print(f"Error: test image does not exist {input_path}")
        return False

    # original text:"\n输入图像: {input_path}"
    print(f"\nInput image: {input_path}")
    # original text:"开始转换..."
    print("Starting conversion...")
    
    start_time = time.time()
    
    try:
        # 使用封装的函数接口
        # Use the wrapped function interface
        output_path = diffv2ir_convert(
            input_path=input_path,
            output_path="test_output/DJI_0061_api_test.png",
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=512,
            steps=30,  # 使用较少步数加快测试 (use fewer steps to speed up the test)
            use_fp16=True,  # 使用半精度 (use half precision)
            seed=42  # 固定种子以便复现 (fixed seed for reproducibility)
        )

        elapsed_time = time.time() - start_time

        # original text:"转换成功！"
        print(f"Conversion succeeded!")
        # original text:"输出路径: {output_path}"
        print(f"Output path: {output_path}")
        # original text:"耗时: {elapsed_time:.2f} 秒"
        print(f"Elapsed time: {elapsed_time:.2f} s")

        # 验证输出文件
        # Verify the output file
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            # original text:"输出文件大小: {file_size:.2f} KB"
            print(f"Output file size: {file_size:.2f} KB")
            return True
        else:
            # original text:"错误：输出文件未创建"
            print("Error: output file was not created")
            return False

    except Exception as e:
        # original text:"转换失败: {str(e)}"
        print(f"Conversion failed: {str(e)}")
        return False


def test_class_api():
    """测试类接口
    Test the class interface
    """
    print("\n" + "=" * 60)
    # original text:"测试类接口 DiffV2IR"
    print("Testing the class interface DiffV2IR")
    print("=" * 60)

    try:
        # 初始化模型
        # Initialize the model
        # original text:"\n初始化 DiffV2IR 模型..."
        print("\nInitializing the DiffV2IR model...")
        model = DiffV2IR(
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            use_fp16=True,
            load_blip=True
        )
        # original text:"模型初始化成功！"
        print("Model initialized successfully!")

        # 测试单张图像
        # Test a single image
        # original text:"\n测试单张图像转换..."
        print("\nTesting single-image conversion...")
        input_path = "data_for_diffv2ir/input/DJI_0068.png"

        start_time = time.time()
        output_path = model.convert(
            input_path=input_path,
            output_path="test_output/DJI_0068_class_test.png",
            resolution=512,
            steps=20,  # 快速测试 (quick test)
            seed=42
        )
        elapsed_time = time.time() - start_time

        # original text:"转换完成: {output_path}"
        print(f"Conversion complete: {output_path}")
        # original text:"耗时: {elapsed_time:.2f} 秒"
        print(f"Elapsed time: {elapsed_time:.2f} s")

        # 测试批量转换
        # Test batch conversion
        # original text:"\n测试批量转换（前3张图像）..."
        print("\nTesting batch conversion (first 3 images)...")
        
        # 创建临时输入文件夹，只包含几张图像
        # Create a temporary input folder containing only a few images
        test_input_dir = "test_input"
        test_output_dir = "test_output/batch"

        os.makedirs(test_input_dir, exist_ok=True)

        # 复制几张测试图像
        # Copy a few test images
        import shutil
        test_images = ["DJI_0061.png", "DJI_0068.png", "DJI_0082.png"]
        for img in test_images:
            src = f"data_for_diffv2ir/input/{img}"
            dst = f"{test_input_dir}/{img}"
            if os.path.exists(src):
                shutil.copy2(src, dst)

        # 执行批量转换
        # Run the batch conversion
        start_time = time.time()
        results = model.batch_convert(
            input_folder=test_input_dir,
            output_folder=test_output_dir,
            resolution=512,
            steps=20,
            seed=42
        )
        elapsed_time = time.time() - start_time

        # original text:"\n批量转换完成！"
        print(f"\nBatch conversion complete!")
        # original text:"处理 {len(results)} 张图像"
        print(f"Processed {len(results)} images")
        # original text:"总耗时: {elapsed_time:.2f} 秒"
        print(f"Total elapsed time: {elapsed_time:.2f} s")
        # original text:"平均每张: {elapsed_time/len(results):.2f} 秒"
        print(f"Average per image: {elapsed_time/len(results):.2f} s")

        # 显示结果
        # Display the results
        for input_name, output_path in results.items():
            if output_path:
                print(f"  ✓ {input_name} -> {output_path}")
            else:
                # original text:"  ✗ {input_name} 转换失败"
                print(f"  ✗ {input_name} conversion failed")

        # 清理临时文件夹
        # Clean up the temporary folder
        shutil.rmtree(test_input_dir)

        return True

    except Exception as e:
        # original text:"测试失败: {str(e)}"
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_output_path():
    """测试自动生成输出路径功能
    Test the automatic output-path generation feature
    """
    print("\n" + "=" * 60)
    # original text:"测试自动输出路径生成"
    print("Testing automatic output-path generation")
    print("=" * 60)

    input_path = "data_for_diffv2ir/input/DJI_0082.png"

    try:
        # original text:"\n输入: {input_path}"
        print(f"\nInput: {input_path}")
        # original text:"不指定输出路径，测试自动生成..."
        print("Not specifying an output path, testing auto-generation...")

        output_path = diffv2ir_convert(
            input_path=input_path,
            # output_path 不指定，会自动生成 (output_path not specified; it will be auto-generated)
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=256,  # 使用较小分辨率加快测试 (use a smaller resolution to speed up the test)
            steps=10,  # 最少步数，只为测试功能 (minimum steps, only to test the feature)
            use_fp16=True,
            seed=42
        )

        # original text:"自动生成的输出路径: {output_path}"
        print(f"Auto-generated output path: {output_path}")

        # 验证文件名格式
        # Verify the filename format
        expected_suffix = "_infrared.png"
        if expected_suffix in output_path:
            # original text:"✓ 输出文件名格式正确（包含 {expected_suffix}）"
            print(f"✓ Output filename format is correct (contains {expected_suffix})")
        else:
            # original text:"✗ 输出文件名格式可能不正确"
            print(f"✗ Output filename format may be incorrect")

        # 清理自动生成的文件
        # Clean up the auto-generated file
        if os.path.exists(output_path):
            os.remove(output_path)
            # original text:"清理测试文件完成"
            print("Test file cleanup complete")

        return True

    except Exception as e:
        # original text:"测试失败: {str(e)}"
        print(f"Test failed: {str(e)}")
        return False


def main():
    """运行所有测试
    Run all the tests
    """
    print("\n" + "=" * 60)
    # original text:"DiffV2IR API 功能测试"
    print("DiffV2IR API functionality test")
    print("=" * 60)

    # 创建输出目录
    # Create the output directories
    os.makedirs("test_output", exist_ok=True)
    os.makedirs("test_output/batch", exist_ok=True)

    # 运行测试
    # Run the tests
    results = []

    # 测试1: 简单函数接口
    # Test 1: simple function interface
    # original text:"\n[测试 1/3]"
    print("\n[Test 1/3]")
    # original text:"简单函数接口"
    results.append(("Simple function interface", test_simple_api()))

    # 测试2: 类接口
    # Test 2: class interface
    # original text:"\n[测试 2/3]"
    print("\n[Test 2/3]")
    # original text:"类接口"
    results.append(("Class interface", test_class_api()))

    # 测试3: 自动输出路径
    # Test 3: automatic output path
    # original text:"\n[测试 3/3]"
    print("\n[Test 3/3]")
    # original text:"自动输出路径"
    results.append(("Automatic output path", test_auto_output_path()))

    # 总结
    # Summary
    print("\n" + "=" * 60)
    # original text:"测试总结"
    print("Test summary")
    print("=" * 60)

    for test_name, success in results:
        # original text:"✓ 通过" / "✗ 失败"
        status = "✓ passed" if success else "✗ failed"
        print(f"{test_name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        # original text:"\n🎉 所有测试通过！"
        print("\n🎉 All tests passed!")
    else:
        # original text:"\n⚠️ 部分测试失败，请检查错误信息"
        print("\n⚠️ Some tests failed, please check the error messages")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)