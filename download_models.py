#!/usr/bin/env python
"""
Script to download all required models to local directories.
This avoids downloading to cache directories during runtime.
"""

import os
import sys
import urllib.request
from pathlib import Path
import argparse

from model_paths import MODEL_PATHS, DOWNLOAD_URLS, check_model_exists

def download_file(url, destination):
    """Download a file with progress bar."""
    def download_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        mb_downloaded = downloaded / 1024 / 1024
        mb_total = total_size / 1024 / 1024
        sys.stdout.write(f'\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=download_progress)
        print()  # New line after progress bar
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download models for DiffV2IR')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Base directory for models (default: models)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip downloading models that already exist')
    parser.add_argument('--models', nargs='+', 
                        choices=['blip', 'clip', 'sam', 'all'],
                        default=['all'],
                        help='Which models to download (default: all)')
    
    args = parser.parse_args()
    
    # Import at function level to handle reloading
    global MODEL_PATHS, DOWNLOAD_URLS, check_model_exists
    
    # Update model base directory if specified
    if args.model_dir != 'models':
        os.environ['DIFFV2IR_MODEL_DIR'] = args.model_dir
        # Reload model paths with new base directory
        from importlib import reload
        import model_paths
        reload(model_paths)
    
    from model_paths import MODEL_PATHS, DOWNLOAD_URLS, check_model_exists
    
    models_to_download = []
    if 'all' in args.models:
        models_to_download = ['blip', 'clip']  # SAM is optional
    else:
        models_to_download = args.models
    
    print(f"Model directory: {args.model_dir}")
    print("-" * 50)
    
    for model_name in models_to_download:
        model_path = MODEL_PATHS.get(model_name)
        download_url = DOWNLOAD_URLS.get(model_name)
        
        if not model_path or not download_url:
            print(f"Unknown model: {model_name}")
            continue
        
        print(f"\nModel: {model_name}")
        print(f"Local path: {model_path}")
        
        if check_model_exists(model_name):
            if args.skip_existing:
                print("✓ Already exists, skipping")
                continue
            else:
                print("✓ Already exists")
                response = input("Re-download? (y/N): ")
                if response.lower() != 'y':
                    continue
        
        print(f"Downloading from: {download_url}")
        success = download_file(download_url, model_path)
        
        if success:
            file_size = os.path.getsize(model_path) / 1024 / 1024
            print(f"✓ Downloaded successfully ({file_size:.1f} MB)")
        else:
            print(f"✗ Failed to download {model_name}")
    
    print("\n" + "=" * 50)
    print("Download summary:")
    for model_name in models_to_download:
        if check_model_exists(model_name):
            print(f"✓ {model_name}: {MODEL_PATHS[model_name]}")
        else:
            print(f"✗ {model_name}: Not found")
    
    # Check for DiffV2IR checkpoint
    diffv2ir_path = MODEL_PATHS.get('diffv2ir')
    if os.path.exists(diffv2ir_path):
        print(f"✓ DiffV2IR checkpoint: {diffv2ir_path}")
    else:
        print(f"\n⚠ DiffV2IR checkpoint not found at: {diffv2ir_path}")
        print("  Please download it manually from:")
        # original text:"  - 夸克网盘: https://pan.quark.cn/s/e2f28304ee90 (访问码: EWCz)"
        print("  - Quark NetDisk: https://pan.quark.cn/s/e2f28304ee90 (access code: EWCz)")
        print("  - HuggingFace: https://huggingface.co/datasets/Lidong26/IR-500K/tree/main")

if __name__ == "__main__":
    main()