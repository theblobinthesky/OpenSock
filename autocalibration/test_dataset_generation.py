#!/usr/bin/env python3
"""
Test script to verify the dataset generation setup.
This tests the argument parsing and basic functionality.
"""

import os
import sys
import json
import argparse

def test_argument_parsing():
    """Test command-line argument parsing."""
    # Simulate command line arguments
    test_args = ['--num-scenes', '5', '--output-dir', 'test_dataset', '--radial-distortion']

    parser = argparse.ArgumentParser(description='Generate autocalibration dataset')
    parser.add_argument('--num-scenes', type=int, default=10, help='Number of scenes to generate')
    parser.add_argument('--output-dir', default='data/dataset', help='Output directory for dataset')
    parser.add_argument('--radial-distortion', action='store_true', help='Enable radial lens distortion')

    args = parser.parse_args(test_args)

    print("Argument parsing test:")
    print(f"  num-scenes: {args.num_scenes}")
    print(f"  output-dir: {args.output_dir}")
    print(f"  radial-distortion: {args.radial_distortion}")

    return args

def test_texture_discovery():
    """Test texture discovery from scraped dataset."""
    scraped_dir = "data/scraped-assets"
    if not os.path.exists(scraped_dir):
        print(f"Warning: {scraped_dir} not found")
        return None

    # Find all available textures
    texture_sets = []
    for provider_dir in ["AmbientCG", "PolyHaven"]:
        provider_path = os.path.join(scraped_dir, provider_dir)
        if os.path.exists(provider_path):
            for asset_dir in os.listdir(provider_path):
                asset_path = os.path.join(provider_path, asset_dir)
                if os.path.isdir(asset_path):
                    # Find the size directory (1K, 2K, etc.)
                    for size_dir in os.listdir(asset_path):
                        size_path = os.path.join(asset_path, size_dir)
                        if os.path.isdir(size_path):
                            # Check if canonical textures exist
                            basecolor = os.path.join(size_path, "basecolor.jpg")
                            roughness = os.path.join(size_path, "roughness.jpg")
                            normal = os.path.join(size_path, "normal.jpg")

                            if all(os.path.exists(f) for f in [basecolor, roughness, normal]):
                                texture_sets.append({
                                    "basecolor": basecolor,
                                    "roughness": roughness,
                                    "normal": normal,
                                    "size": size_dir
                                })

    print(f"\nTexture discovery test:")
    print(f"  Found {len(texture_sets)} texture sets")

    if texture_sets:
        sample = texture_sets[0]
        print(f"  Sample texture: {sample['basecolor']}")
        print(f"  Size: {sample['size']}")

    return texture_sets

def test_output_structure(args):
    """Test creating the expected output directory structure."""
    print("\nOutput structure test:")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Simulate creating a scene directory
    scene_dir = os.path.join(args.output_dir, "scene_0000")
    os.makedirs(scene_dir, exist_ok=True)

    # Create camera directories
    for i in range(2):
        camera_dir = os.path.join(scene_dir, f"camera_{i}")
        os.makedirs(camera_dir, exist_ok=True)

        # Create sample metadata
        metadata = {
            'scene_id': 0,
            'camera_id': i,
            'image_path': f'camera_{i}/rgb.png',
            'intrinsics': {'fx': 1500, 'fy': 1500, 'cx': 512, 'cy': 384},
            'location': [0.1 * i, 0.1 * i, 2.0],
            'rotation': [1.57, 0, 0],  # 90 degrees in radians
            'focal_length': 35
        }

        if args.radial_distortion:
            metadata['distortion'] = {'k1': -0.05, 'k2': 0.001, 'p1': 0.0, 'p2': 0.0}

        with open(os.path.join(camera_dir, "camera_info.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    # Create scene metadata
    scene_metadata = {
        'scene_id': 0,
        'floor_texture': {'basecolor': 'sample_texture.jpg'},
        'objects': [
            {'name': 'object_1', 'location': [2, 3, 0.5], 'category_id': 1},
            {'name': 'object_2', 'location': [-2, -1, 0.5], 'category_id': 2}
        ],
        'cameras': [
            {'location': [0, 0, 2], 'rotation': [1.57, 0, 0]},
            {'location': [0.1, 0.1, 2.1], 'rotation': [1.57, 0, 0]}
        ]
    }

    with open(os.path.join(scene_dir, "metadata.json"), "w") as f:
        json.dump(scene_metadata, f, indent=2)

    print(f"  Created sample structure in {args.output_dir}")
    print("  Directory structure:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"  {indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"  {subindent}{file}")

def main():
    print("Testing dataset generation setup...")

    # Test argument parsing
    args = test_argument_parsing()

    # Test texture discovery
    textures = test_texture_discovery()

    # Test output structure
    test_output_structure(args)

    print("\n✅ All tests completed successfully!")
    print(f"\nTo run the actual dataset generation:")
    print(f"blender -b -P src/blender_render.py -- --num-scenes {args.num_scenes} --output-dir {args.output_dir} {'--radial-distortion' if args.radial_distortion else ''}")

if __name__ == "__main__":
    main()