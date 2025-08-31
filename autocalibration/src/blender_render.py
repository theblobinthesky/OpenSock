# run with: blender -b -P blender_render.py
#
# Modified Blender renderer for autocalibration pipeline
# Features:
# - Uses scraped floor textures from AmbientCG/PolyHaven
# - Creates repeating floor plane at (0,0)
# - Top-down camera view for keypoint correspondence
# - Renders two images from similar viewpoints
# - Randomly places objects from data/models directory
# - Physics simulation for natural object placement
# - Generates entire datasets with configurable size
#
import bpy
from mathutils import Vector
import numpy as np
import random
import math
import json
import os
import sys
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate autocalibration dataset')
parser.add_argument('--num-scenes', type=int, default=10, help='Number of scenes to generate')
parser.add_argument('--output-dir', default='data/dataset', help='Output directory for dataset')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])

# Set up basic scene
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32  # Increased for better quality
scene.cycles.use_denoising = False

# Set render resolution
scene.render.resolution_x = 1024  # Reduced for faster generation
scene.render.resolution_y = 768
scene.render.resolution_percentage = 100

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Function to get random floor texture from scraped dataset
def get_random_floor_texture():
    """Get a random floor texture from the scraped dataset."""
    scraped_dir = "data/scraped-assets"
    if not os.path.exists(scraped_dir):
        print(f"Warning: {scraped_dir} not found, using default textures")
        return {
            "basecolor": "textures/floor/albedo.jpg",
            "roughness": "textures/floor/roughness.jpg",
            "normal": "textures/floor/normal.jpg",
            "displacement": "textures/floor/height.exr",
            "ao": "textures/floor/ao.jpg"
        }

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
        print("Warning: No texture sets found, using defaults")
        return {
            "basecolor": "textures/floor/albedo.jpg",
            "roughness": "textures/floor/roughness.jpg",
            "normal": "textures/floor/normal.jpg",
            "displacement": "textures/floor/height.exr",
            "ao": "textures/floor/ao.jpg"
        }

    return random.choice(texture_sets)

# Create repeating floor plane
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))  # Large plane for repeating texture
floor = bpy.context.active_object
floor.name = "Floor"

# Get random floor texture
floor_textures = get_random_floor_texture()

# Create floor material
mat = bpy.data.materials.new(name='floor_mat')
mat.use_nodes = True

# Get Principled BSDF node
principled_bsdf = None
for node in mat.node_tree.nodes:
    if node.type == 'BSDF_PRINCIPLED':
        principled_bsdf = node
        break

if not principled_bsdf:
    principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

# Try to set basic texture if available
if os.path.exists(floor_textures["basecolor"]):
    try:
        # Create image texture node
        tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex_node.image = bpy.data.images.load(floor_textures["basecolor"])

        # Link to base color
        mat.node_tree.links.new(tex_node.outputs['Color'], principled_bsdf.inputs['Base Color'])

        # Set UV scaling for repetition
        mapping_node = mat.node_tree.nodes.new('ShaderNodeMapping')
        uv_node = mat.node_tree.nodes.new('ShaderNodeTexCoord')

        mapping_node.inputs['Scale'].default_value = (5, 5, 1)  # 5x repetition

        mat.node_tree.links.new(uv_node.outputs['UV'], mapping_node.inputs['Vector'])
        mat.node_tree.links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])

        print(f"Using floor texture: {floor_textures['basecolor']}")
    except Exception as e:
        print(f"Warning: Could not set texture: {e}, using default material")

# Assign material to floor
if floor.data.materials:
    floor.data.materials[0] = mat
else:
    floor.data.materials.append(mat)

# Load random objects from data/models directory
def load_random_objects():
    """Load random objects from data/models directory."""
    models_dir = "data/models"
    if not os.path.exists(models_dir):
        print(f"Warning: {models_dir} not found, skipping object loading")
        return []

    # Find all OBJ files in the models directory
    obj_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.lower().endswith('.obj'):
                obj_files.append(os.path.join(root, file))

    if not obj_files:
        print(f"Warning: No OBJ files found in {models_dir}")
        return []
    print(f"{obj_files=}")

    # Load 2-5 random objects
    num_objects = random.randint(5, 15)
    selected_files = random.sample(obj_files, min(num_objects, len(obj_files)))

    objects = []
    for i, obj_file in enumerate(selected_files):
        try:
            # Load the object using Blender's import
            bpy.ops.wm.obj_import(filepath=obj_file)
            obj = bpy.context.selected_objects[0]  # Get the imported object

            # Set custom property for category ID
            obj["category_id"] = i + 1

            # Random rotation
            obj.rotation_euler = (
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi)
            )

            # Random position on floor (within the plane bounds)
            x = random.uniform(-1, 1) * 0.5
            y = random.uniform(-1, 1) * 0.5
            z = 0.5  # Slightly above floor to avoid clipping
            obj.location = (x, y, z)

            # Random scale (0.5x to 2x)
            scale = random.uniform(0.5, 2.0) * 0.05
            obj.scale = (scale, scale, scale)

            objects.append(obj)
            print(f"Loaded object: {obj_file} at ({x:.1f}, {y:.1f}, {z:.1f})")
        except Exception as e:
            print(f"Error loading {obj_file}: {e}")

    return objects

# Load random objects
random_objects = load_random_objects()

# Simple physics simulation for natural poses (only if we have objects)
if random_objects:
    # Enable rigid body physics
    bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.time_scale = 1.0

    # Add rigid body to objects
    for obj in random_objects:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()

    # Bake physics
    bpy.ops.ptcache.bake_all(bake=True)

    # Remove rigid body components after baking
    for obj in random_objects:
        if obj.rigid_body:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.rigidbody.object_remove()

# Camera setup for top-down view of floor
def sample_camera_pair():
    """Generate two cameras with similar viewpoints for keypoint correspondence."""
    cameras = []

    # Base position and orientation (looking down at floor)
    base_height = 1.8  # Height above floor
    base_look_at = np.array([0, 0, 0])  # Look at center of floor

    for _ in range(2):
        # Small random variation in position for the two cameras
        # Keep them close together for keypoint correspondence
        x_offset = np.random.uniform(-0.5, 0.5)
        y_offset = np.random.uniform(-0.5, 0.5)
        height_variation = np.random.uniform(-0.2, 0.2)

        location = np.array([
            x_offset,
            y_offset,
            base_height + height_variation
        ])

        # Calculate rotation so camera looks at base_look_at
        # Blender cameras look along -Z with +Y as up
        try:
            target_vec = Vector(base_look_at.tolist())
            loc_vec = Vector(location.tolist())
            direction = target_vec - loc_vec
            if direction.length == 0:
                rotation = np.array([0.0, 0.0, 0.0])
            else:
                rot_euler = direction.to_track_quat('-Z', 'Y').to_euler('XYZ')
                rotation = np.array([rot_euler.x, rot_euler.y, rot_euler.z])
        except Exception as e:
            print(f"Warning computing look-at rotation: {e}")
            rotation = np.array([0.0, 0.0, 0.0])

        # Phone-like camera settings
        focal_length = np.random.uniform(22, 26)

        # Add radial distortion if enabled
        distortion_coeffs = {
            'k1': np.random.uniform(-0.1, 0.1),
            'k2': np.random.uniform(-0.01, 0.01),
            'p1': 0.0,
            'p2': 0.0
        }

        camera_data = {
            'location': location,
            'rotation': rotation,
            'focal_length': focal_length,
            'fov': np.deg2rad(45),  # Approximate FOV
            'intrinsics': {
                'fx': 1500,  # Approximate focal length in pixels
                'fy': 1500,
                'cx': 512,   # Principal point
                'cy': 384
            },
            'distortion': distortion_coeffs
        }

        cameras.append(camera_data)

    return cameras

# Main dataset generation loop
print(f"Generating {args.num_scenes} scenes...")

for scene_idx in range(args.num_scenes):
    print(f"\n=== Generating Scene {scene_idx + 1}/{args.num_scenes} ===")

    # Clear scene for new generation
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Create floor
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"

    # Get random floor texture
    floor_textures = get_random_floor_texture()

    # Create floor material
    mat = bpy.data.materials.new(name=f'floor_mat_{scene_idx}')
    mat.use_nodes = True

    # Get Principled BSDF node
    principled_bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break

    if not principled_bsdf:
        principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

    # Try to set basic texture if available
    if os.path.exists(floor_textures["basecolor"]):
        try:
            # Create image texture node
            tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
            tex_node.image = bpy.data.images.load(floor_textures["basecolor"])

            # Link to base color
            mat.node_tree.links.new(tex_node.outputs['Color'], principled_bsdf.inputs['Base Color'])

            # Set UV scaling for repetition
            mapping_node = mat.node_tree.nodes.new('ShaderNodeMapping')
            uv_node = mat.node_tree.nodes.new('ShaderNodeTexCoord')
            mapping_node.inputs['Scale'].default_value = (5, 5, 1)  # 5x repetition

            mat.node_tree.links.new(uv_node.outputs['UV'], mapping_node.inputs['Vector'])
            mat.node_tree.links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])

            print(f"Using floor texture: {floor_textures['basecolor']}")
        except Exception as e:
            print(f"Warning: Could not set texture: {e}, using default material")

    # Assign material to floor
    if floor.data.materials:
        floor.data.materials[0] = mat
    else:
        floor.data.materials.append(mat)

    # Load random objects
    random_objects = load_random_objects()

    # Generate two cameras with similar viewpoints
    cameras = sample_camera_pair()

    # Create camera objects in Blender
    blender_cameras = []
    for i, cam_data in enumerate(cameras):
        # Create camera
        bpy.ops.object.camera_add(location=cam_data['location'])
        cam_obj = bpy.context.active_object
        cam_obj.name = f"Camera_{i}"

        # Set rotation
        cam_obj.rotation_euler = cam_data['rotation']

        # Set camera properties
        cam = cam_obj.data
        cam.lens = cam_data['focal_length']
        cam.sensor_width = 36.0  # Full frame

        # Add distortion if enabled
        if cam_data['distortion']:
            # Note: Blender doesn't have built-in radial distortion
            # This would require additional camera post-processing
            pass

        blender_cameras.append(cam_obj)

    # Set the first camera as active
    scene.camera = blender_cameras[0]

    # Lighting: Simple setup for top-down view
    # Add a sun light for consistent floor lighting
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = 5.0
    sun.data.angle = np.deg2rad(45)  # Soft shadows

    # Set world background to neutral gray
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)  # Neutral gray
    bg_node.inputs['Strength'].default_value = 1.0

    # Render two images from different camera positions
    print(f"Rendering scene {scene_idx + 1}...")

    # Create scene-specific output directory
    scene_output_dir = os.path.join(args.output_dir, f"scene_{scene_idx:04d}")
    os.makedirs(scene_output_dir, exist_ok=True)

    # Render from both cameras
    scene_images = []
    for i, cam_obj in enumerate(blender_cameras):
        print(f"  Rendering camera {i+1}/2...")

        # Set camera for this render
        scene.camera = cam_obj

        # Set output path for this camera
        camera_output_dir = os.path.join(scene_output_dir, f"camera_{i}")
        os.makedirs(camera_output_dir, exist_ok=True)

        # Set render output
        scene.render.filepath = os.path.join(camera_output_dir, "rgb.png")

        # Render the image
        bpy.ops.render.render(write_still=True)

        scene_images.append({
            'camera_id': i,
            'image_path': os.path.join(camera_output_dir, "rgb.png"),
            'camera_data': cameras[i]
        })

        print(f"  Completed camera {i+1}/2")

    # Save scene metadata
    scene_metadata = {
        'scene_id': scene_idx,
        'floor_texture': floor_textures,
        'objects': [{'name': obj.name, 'location': list(obj.location), 'rotation': list(obj.rotation_euler), 'scale': list(obj.scale), 'category_id': obj.get('category_id', 0)} for obj in random_objects],
        'cameras': cameras,
        'images': scene_images
    }

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open(os.path.join(scene_output_dir, "metadata.json"), "w") as f:
        print(f"{scene_metadata=}")
        json.dump(scene_metadata, f, indent=2, cls=NumpyEncoder)

    print(f"Completed scene {scene_idx + 1}")

print(f"\nDataset generation completed!")
print(f"Generated {args.num_scenes} scenes in {args.output_dir}")
print("Each scene contains:")
print("  - 2 camera views with metadata")
print("  - Random floor texture")
print("  - Random object placement")
print("  - Camera intrinsics and poses")
