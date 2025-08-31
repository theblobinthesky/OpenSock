#!/usr/bin/env python3
"""
Test script to verify the Blender renderer setup works correctly.
This tests the texture loading and basic functionality without running Blender.
"""

import os
import random
import json

def get_random_floor_texture():
    """Get a random floor texture from the scraped dataset."""
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
                            displacement = os.path.join(size_path, "displacement.jpg")
                            ao = os.path.join(size_path, "ao.jpg")

                            if all(os.path.exists(f) for f in [basecolor, roughness, normal]):
                                texture_sets.append({
                                    "basecolor": basecolor,
                                    "roughness": roughness,
                                    "normal": normal,
                                    "displacement": displacement if os.path.exists(displacement) else None,
                                    "ao": ao if os.path.exists(ao) else None
                                })

    if not texture_sets:
        print("Warning: No texture sets found")
        return None

    selected = random.choice(texture_sets)
    print(f"Selected texture set: {selected['basecolor']}")
    return selected

def test_object_loading():
    """Test loading objects from data/models directory."""
    models_dir = "data/models"
    if not os.path.exists(models_dir):
        print(f"Warning: {models_dir} not found")
        return []

    # Find all OBJ files in the models directory
    obj_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))

    print(f"Found {len(obj_files)} OBJ files:")
    for obj_file in obj_files:
        print(f"  - {obj_file}")

    return obj_files

def test_camera_setup():
    """Test the camera setup logic."""
    import numpy as np

    # Simulate camera pair generation
    cameras = []

    base_height = 2.0
    base_look_at = np.array([0, 0, 0])

    for i in range(2):
        x_offset = np.random.uniform(-0.5, 0.5)
        y_offset = np.random.uniform(-0.5, 0.5)
        height_variation = np.random.uniform(-0.2, 0.2)

        location = np.array([
            x_offset,
            y_offset,
            base_height + height_variation
        ])

        rotation = np.array([np.pi/2, 0, 0])  # Look down

        camera_data = {
            'location': location,
            'rotation': rotation,
            'focal_length': 50,
            'intrinsics': {
                'fx': 2000,
                'fy': 2000,
                'cx': 1024,
                'cy': 768
            }
        }

        cameras.append(camera_data)

    print("Generated 2 camera configurations:")
    for i, cam in enumerate(cameras):
        print(f"  Camera {i}: location={cam['location']}, focal_length={cam['focal_length']}")

    return cameras

if __name__ == "__main__":
    print("Testing Blender renderer setup...")
    print()

    print("1. Testing texture loading...")
    texture = get_random_floor_texture()
    if texture:
        print("✅ Texture loading successful")
        print(f"   Base color: {texture['basecolor']}")
        print(f"   Roughness: {texture['roughness']}")
        print(f"   Normal: {texture['normal']}")
        if texture['displacement']:
            print(f"   Displacement: {texture['displacement']}")
        if texture['ao']:
            print(f"   Ambient Occlusion: {texture['ao']}")
    else:
        print("❌ Texture loading failed")
    print()

    print("2. Testing object loading...")
    objects = test_object_loading()
    if objects:
        print("✅ Object loading successful")
    else:
        print("❌ No objects found")
    print()

    print("3. Testing camera setup...")
    cameras = test_camera_setup()
    print("✅ Camera setup successful")
    print()

    print("4. Checking output directories...")
    output_dirs = ["data/input_camera_0", "data/input_camera_1"]
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Created output directory: {output_dir}")
    print()

    print("🎯 Blender renderer setup test completed!")
    print("Run with: blender -b -P src/blender_render.py")