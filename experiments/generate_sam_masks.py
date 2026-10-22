#!/usr/bin/env python3
"""
Generate SAM masks for xmu dataset to be used with DiffV2IR
Based on SAM automatic mask generation
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json

def load_sam_model(model_type="vit_h", checkpoint_path="SAM_models/sam_vit_h_4b8939.pth"):
    """Load SAM model"""
    print(f"Loading SAM model: {model_type} from {checkpoint_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return sam

def generate_combined_mask(image_path, sam_model, output_path=None):
    """
    Generate a combined mask for the entire image using SAM
    Returns a binary mask where objects are white (255) and background is black (0)
    """
    print(f"Processing: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate masks using SAM
    mask_generator = SamAutomaticMaskGenerator(
        sam_model,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires at least 100 pixels
    )
    
    masks = mask_generator.generate(image_rgb)
    print(f"Generated {len(masks)} masks")
    
    # Create combined mask
    h, w = image_rgb.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Sort masks by area (largest first) and combine them
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    for mask_dict in masks:
        mask = mask_dict['segmentation']
        # Add mask to combined mask (white for objects)
        combined_mask[mask] = 255
    
    # Save mask
    if output_path:
        cv2.imwrite(output_path, combined_mask)
        print(f"Saved mask to: {output_path}")
    
    return combined_mask

def process_directory(input_dir, output_dir, model_type="vit_h", checkpoint_path="SAM_models/sam_vit_h_4b8939.pth"):
    """Process all images in a directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load SAM model
    sam_model = load_sam_model(model_type, checkpoint_path)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext.lower())])
    
    image_files.sort()
    print(f"Found {len(image_files)} images")
    
    # Process each image
    results = {}
    for i, image_file in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {image_file}")
        
        input_path = os.path.join(input_dir, image_file)
        if input_dir==output_dir:
            output_file = os.path.splitext(image_file)[0] + '_mask.png'
        else:
            output_file = os.path.splitext(image_file)[0] + '.png'
            
        output_path = os.path.join(output_dir, output_file)
        
        try:
            mask = generate_combined_mask(input_path, sam_model, output_path)
            results[image_file] = {
                'success': True,
                'mask_file': output_file,
                'white_pixels': int(np.sum(mask == 255)),
                'total_pixels': int(mask.size),
                'coverage': float(np.sum(mask == 255) / mask.size)
            }
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results[image_file] = {
                'success': False,
                'error': str(e)
            }
    
    # Save results summary
    results_file = os.path.join(output_dir, 'mask_generation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCompleted! Results saved to {results_file}")
    
    # Print summary
    successful = sum(1 for r in results.values() if r['success'])
    print(f"Successfully processed: {successful}/{len(image_files)} images")
    
    if successful > 0:
        avg_coverage = np.mean([r['coverage'] for r in results.values() if r['success']])
        print(f"Average mask coverage: {avg_coverage:.1%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate SAM masks for images')
    parser.add_argument('--input', required=True, help='Input directory with images')
    parser.add_argument('--output', required=True, help='Output directory for masks')
    parser.add_argument('--model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], 
                        help='SAM model type')
    parser.add_argument('--checkpoint', default='SAM_models/sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output, args.model_type, args.checkpoint)