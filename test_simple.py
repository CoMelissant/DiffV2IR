"""
简单测试 DiffV2IR API
Simple test of the DiffV2IR API
"""

import os
import sys
sys.path.append('/ssd3/zhiwen/projects/z_workspace/TIR_3DGS/DiffV2IR')

def test_import():
    """测试导入
    Test the import
    """
    try:
        from diffv2ir_api import diffv2ir_convert, DiffV2IR
        # original text:"✓ 成功导入 diffv2ir_api"
        print("✓ Successfully imported diffv2ir_api")
        return True
    except Exception as e:
        # original text:"✗ 导入失败: {e}"
        print(f"✗ Import failed: {e}")
        return False

def test_basic_convert():
    """测试基本转换功能
    Test the basic conversion functionality
    """
    from diffv2ir_api import diffv2ir_convert

    input_path = "data_for_diffv2ir/input/DJI_0061.png"
    output_path = "test_output_simple.png"

    if not os.path.exists(input_path):
        # original text:"测试图像不存在: {input_path}"
        print(f"Test image does not exist: {input_path}")
        return False

    # original text:"测试图像: {input_path}"
    print(f"Test image: {input_path}")

    try:
        # 创建输出目录
        # Create the output directory
        os.makedirs("test_output", exist_ok=True)

        # original text:"开始转换（使用最少步数进行快速测试）..."
        print("Starting conversion (using the minimum steps for a quick test)...")
        result = diffv2ir_convert(
            input_path=input_path,
            output_path=output_path,
            config_path="configs/generate.yaml",
            checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
            resolution=256,  # 低分辨率 (low resolution)
            steps=5,  # 极少步数，只为测试功能 (very few steps, only to test the feature)
            use_fp16=True,
            seed=42
        )

        # original text:"✓ 转换成功！输出: {result}"
        print(f"✓ Conversion succeeded! Output: {result}")

        # 验证文件
        # Verify the file
        if os.path.exists(result):
            size = os.path.getsize(result) / 1024
            # original text:"✓ 输出文件存在，大小: {size:.2f} KB"
            print(f"✓ Output file exists, size: {size:.2f} KB")
            # 不删除文件，保留用于检查
            # Do not delete the file; keep it for inspection
            # os.remove(result)
            return True
        else:
            # original text:"✗ 输出文件不存在"
            print("✗ Output file does not exist")
            return False

    except Exception as e:
        # original text:"✗ 转换失败: {e}"
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    # original text:"DiffV2IR API 简单测试"
    print("DiffV2IR API simple test")
    print("=" * 60)

    # 测试导入
    # Test the import
    # original text:"\n[1] 测试导入..."
    print("\n[1] Testing import...")
    success1 = test_import()

    # 测试基本功能
    # Test the basic functionality
    # original text:"\n[2] 测试基本转换功能..."
    print("\n[2] Testing basic conversion...")
    success2 = test_basic_convert()

    # 总结
    # Summary
    print("\n" + "=" * 60)
    if success1 and success2:
        # original text:"✓ 测试通过！API 可正常使用"
        print("✓ Test passed! The API works normally")
    else:
        # original text:"✗ 测试失败，请检查错误信息"
        print("✗ Test failed, please check the error messages")