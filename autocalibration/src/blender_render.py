import bpy
from mathutils import Vector
import numpy as np
import random
import math
import json
import os
import shutil
import sys
import argparse
import glob

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate autocalibration dataset')
parser.add_argument('--num-scenes', type=int, default=1, help='Number of scenes to generate')
parser.add_argument('--output-dir', default=os.path.join(os.path.dirname(__file__), "..", "data", "dataset"), help='Output directory for dataset')
parser.add_argument('--keep-rigidbody', action='store_true', help='Keep rigid body components after sim for inspection')
parser.add_argument('--save-blend', action='store_true', help='Save a .blend file per scene for inspection')
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
    scraped_dir = os.path.join(os.path.dirname(__file__), "..", "data", "scraped-assets")
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


def get_random_hdri():
    hdri_dir = os.path.join(os.path.dirname(__file__), "..", "data", "hdris")
    if not os.path.exists(hdri_dir):
        print(f"Warning: {hdri_dir} not found, using default background")
        return None

    # Find all HDRI files (common formats)
    hdri_files = []
    for ext in [".hdr", ".exr", ".png", ".jpg", ".jpeg"]:
        hdri_files.extend(glob.glob(os.path.join(hdri_dir, f"**/*{ext}"), recursive=True))

    if not hdri_files:
        print("Warning: No HDRI files found, using default background")
        return None

    return random.choice(hdri_files)

def assign_floor_material(floor, floor_textures):
    """Create and assign a repeating textured material to the given floor object."""
    mat = bpy.data.materials.new(name='floor_mat')
    mat.use_nodes = True

    principled_bsdf = None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_bsdf = node
            break

    if not principled_bsdf:
        principled_bsdf = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')

    if floor_textures and os.path.exists(floor_textures.get("basecolor", "")):
        try:
            tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
            tex_node.image = bpy.data.images.load(floor_textures["basecolor"])

            mat.node_tree.links.new(tex_node.outputs['Color'], principled_bsdf.inputs['Base Color'])

            mapping_node = mat.node_tree.nodes.new('ShaderNodeMapping')
            uv_node = mat.node_tree.nodes.new('ShaderNodeTexCoord')
            mapping_node.inputs['Scale'].default_value = (10, 10, 1)  # repeat

            mat.node_tree.links.new(uv_node.outputs['UV'], mapping_node.inputs['Vector'])
            mat.node_tree.links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])

            print(f"Using floor texture: {floor_textures['basecolor']}")
        except Exception as e:
            print(f"Warning: Could not set texture: {e}, using default material")

    if floor.data.materials:
        floor.data.materials[0] = mat
    else:
        floor.data.materials.append(mat)

    return mat

# Real-life sizes in meters (approximate)
real_sizes = {
    "keys": (0.07, 0.07, 0.07),  # 7cm cube
    "pen": (0.14, 0.008, 0.008),  # 14cm long, 8mm diameter
    "credit-card": (0.0856, 0.05398, 0.00076)  # Standard credit card
}

# Load random objects from data/models directory
def load_random_objects():
    """Load random objects from data/models directory."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models")
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
    num_objects = random.randint(3, 7)
    selected_files = random.sample(obj_files, min(num_objects, len(obj_files)))

    objects = []
    for i, obj_file in enumerate(selected_files):
        try:
            # Load the object using Blender's import
            bpy.ops.wm.obj_import(filepath=obj_file)
            selected = bpy.context.selected_objects

            if not selected:
                continue

            # If multiple objects, join them into one
            if len(selected) > 1:
                bpy.ops.object.select_all(action='DESELECT')
                for obj in selected:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = selected[0]
                bpy.ops.object.join()
                selected = [selected[0]]

            obj = selected[0]

            # Update the scene to ensure dimensions are correct
            bpy.context.view_layer.update()

            # Determine object type from path
            obj_type = None
            if "keys" in obj_file.lower():
                obj_type = "keys"
            elif "pen" in obj_file.lower():
                obj_type = "pen"
            elif "credit-card" in obj_file.lower():
                obj_type = "credit-card"

            # Scale to real-life size
            if obj_type and obj_type in real_sizes:
                real_dim = real_sizes[obj_type]
                current_dim = obj.dimensions
                print(f"Current dimensions for {obj_file}: {current_dim}")
                # Calculate scale factor (average for balance)
                scale_factors = [real_dim[i] / current_dim[i] if current_dim[i] > 0 else 1 for i in range(3)]
                scale_factor = sum(scale_factors) / 3
                obj.scale = (scale_factor, scale_factor, scale_factor)
                print(f"Scaled {obj_file} to real size: {scale_factor}, new dimensions: {obj.dimensions}")
            else:
                # Fallback random scale
                scale = random.uniform(0.5, 2.0) * 0.05
                obj.scale = (scale, scale, scale)

            # Set custom property for category ID
            obj["category_id"] = i + 1

            # Random rotation
            obj.rotation_euler = (
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi),
                random.uniform(0, 2*math.pi)
            )

            # Random position on floor (within the plane bounds)
            x = random.uniform(-1, 1) * 1.0
            y = random.uniform(-1, 1) * 1.0
            z = 0.5  # Slightly above floor to avoid clipping
            obj.location = (x, y, z)

            objects.append(obj)
            print(f"Loaded object: {obj_file} at ({x:.1f}, {y:.1f}, {z:.1f})")
        except Exception as e:
            print(f"Error loading {obj_file}: {e}")

    return objects

def run_rigidbody_drop(floor, objects, frame_end: int, keep: bool):
    """Run rigid body simulation so objects settle on the floor (headless-safe)."""
    if not objects:
        return

    scn = bpy.context.scene
    scn.frame_start = 1
    scn.frame_end = frame_end

    print(f"[RB] Preparing rigid body world (objects={len(objects)})")
    # Ensure world exists
    if not scn.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    rbw = scn.rigidbody_world
    # World config with version guards
    if hasattr(rbw, 'time_scale'):
        rbw.time_scale = 1.0
    # Steps per second / substeps naming varies across versions
    if hasattr(rbw, 'steps_per_second'):
        rbw.steps_per_second = 120
    elif hasattr(rbw, 'substeps_per_frame'):
        rbw.substeps_per_frame = 10
    # Solver iterations naming varies
    if hasattr(rbw, 'solver_iterations'):
        rbw.solver_iterations = 25
    elif hasattr(rbw, 'num_solver_iterations'):
        rbw.num_solver_iterations = 25
    scn.use_gravity = True
    if hasattr(scn, 'gravity'):
        scn.gravity = (0.0, 0.0, -9.81)
    print(f"[RB] World configured: time_scale={getattr(rbw,'time_scale',None)}")

    # Make sure the RBW collection includes our objects
    if rbw.collection is None:
        # Create a dedicated collection if missing
        coll = bpy.data.collections.new("RigidBodyWorld")
        bpy.context.scene.collection.children.link(coll)
        rbw.collection = coll
    else:
        coll = rbw.collection

    # Floor as passive collider
    bpy.context.view_layer.objects.active = floor
    if not getattr(floor, 'rigid_body', None):
        bpy.ops.rigidbody.object_add(type='PASSIVE')
    else:
        floor.rigid_body.type = 'PASSIVE'
    floor.rigid_body.use_margin = True
    floor.rigid_body.collision_margin = 0.05
    floor.rigid_body.collision_shape = 'MESH'
    # Link to RBW collection
    try:
        if floor not in coll.objects:
            coll.objects.link(floor)
    except Exception:
        pass
    print(f"[RB] Floor rigid_body: type={floor.rigid_body.type if floor.rigid_body else None}, shape={floor.rigid_body.collision_shape if floor.rigid_body else None}")

    # Active rigid bodies for objects
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        if not getattr(obj, 'rigid_body', None):
            bpy.ops.rigidbody.object_add(type='ACTIVE')
        else:
            obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.mass = 1.0
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.01
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        # Ensure in RBW collection
        try:
            if obj not in coll.objects:
                coll.objects.link(obj)
        except Exception:
            pass
        print(f"[RB] Added ACTIVE rigid body to {obj.name}")

    # Clear any existing cache and step through frames (headless-safe)
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception as e:
        print(f"[RB] free_bake_all failed: {e}")

    # Configure cache range then bake, which works in background mode
    if rbw.point_cache:
        rbw.point_cache.frame_start = scn.frame_start
        rbw.point_cache.frame_end = scn.frame_end
    try:
        bpy.ops.ptcache.bake_all(bake=True)
        print("[RB] Bake success")
    except Exception as e:
        print(f"[RB] Bake failed ({e}), stepping frames manually")
        deps = bpy.context.evaluated_depsgraph_get()
        for f in range(scn.frame_start, scn.frame_end + 1):
            scn.frame_set(f)
            deps.update()

    # Set to final frame to keep settled transforms
    scn.frame_set(scn.frame_end)
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    # Copy evaluated transforms to real objects to preserve pose even if RB removed
    for obj in objects:
        try:
            eval_obj = obj.evaluated_get(deps)
            obj.location = eval_obj.location
            try:
                obj.rotation_euler = eval_obj.rotation_euler
            except Exception:
                pass
        except Exception:
            pass
    for obj in objects:
        print(f"[RB] Settled {obj.name} loc={tuple(round(x,4) for x in obj.location)}")

    # Optionally remove rigid body components after settling
    if not keep:
        for obj in objects:
            if getattr(obj, 'rigid_body', None):
                bpy.context.view_layer.objects.active = obj
                try:
                    bpy.ops.rigidbody.object_remove()
                except Exception:
                    pass
        if getattr(floor, 'rigid_body', None):
            bpy.context.view_layer.objects.active = floor
            try:
                bpy.ops.rigidbody.object_remove()
            except Exception:
                pass

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
shutil.rmtree(args.output_dir, ignore_errors=True)
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

    # Get random floor texture and assign material
    floor_textures = get_random_floor_texture()
    assign_floor_material(floor, floor_textures)

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

    # Set up world background with HDRI or fallback to neutral gray
    world = bpy.context.scene.world
    world.use_nodes = True

    # Clear existing nodes except World Output
    for node in list(world.node_tree.nodes):
        if node.type != 'OUTPUT_WORLD':
            world.node_tree.nodes.remove(node)

    # Get random HDRI
    hdri_path = get_random_hdri()

    if hdri_path:
        # Set up HDRI background
        try:
            # Create Environment Texture node
            env_tex_node = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
            env_tex_node.image = bpy.data.images.load(hdri_path)

            # Create Background node
            bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
            bg_node.inputs['Strength'].default_value = 1.0

            # Create World Output node (should already exist)
            world_output = world.node_tree.nodes.get('World Output')
            if not world_output:
                world_output = world.node_tree.nodes.new('ShaderNodeOutputWorld')

            # Connect nodes: Environment Texture -> Background -> World Output
            world.node_tree.links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
            world.node_tree.links.new(bg_node.outputs['Background'], world_output.inputs['Surface'])

            print(f"Using HDRI: {os.path.basename(hdri_path)}")
        except Exception as e:
            print(f"Warning: Could not load HDRI {hdri_path}: {e}, using default background")
            # Fallback to neutral gray
            bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
            bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
            bg_node.inputs['Strength'].default_value = 1.0

            world_output = world.node_tree.nodes.get('World Output')
            if not world_output:
                world_output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
            world.node_tree.links.new(bg_node.outputs['Background'], world_output.inputs['Surface'])
    else:
        # No HDRI found, use neutral gray background
        bg_node = world.node_tree.nodes.new('ShaderNodeBackground')
        bg_node.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1.0)
        bg_node.inputs['Strength'].default_value = 1.0

        world_output = world.node_tree.nodes.get('World Output')
        if not world_output:
            world_output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
        world.node_tree.links.new(bg_node.outputs['Background'], world_output.inputs['Surface'])

    # Run rigid body sim so objects settle before rendering
    run_rigidbody_drop(floor, random_objects, frame_end=152, keep=args.keep_rigidbody)

    # Prepare output dirs
    camera_0_dir = os.path.join(args.output_dir, "camera_0")
    camera_1_dir = os.path.join(args.output_dir, "camera_1")
    camera_info_dir = os.path.join(args.output_dir, "camera_info")
    blend_dir = os.path.join(args.output_dir, "blend_files")
    os.makedirs(camera_0_dir, exist_ok=True)
    os.makedirs(camera_1_dir, exist_ok=True)
    os.makedirs(camera_info_dir, exist_ok=True)
    os.makedirs(blend_dir, exist_ok=True)

    # Optionally save .blend for inspection (after sim setup)
    if args.save_blend:
        blend_path = os.path.join(blend_dir, f"scene_{scene_idx:04d}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True)

    # Render from each camera and save metadata
    for i, cam_obj in enumerate(blender_cameras):
        cam_dir = camera_0_dir if i == 0 else camera_1_dir

        scene.camera = cam_obj
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.join(cam_dir, f'scene_{scene_idx:04d}_rgb.png')
        bpy.ops.render.render(write_still=True)

        cam_meta = {
            'scene_id': scene_idx,
            'camera_id': i,
            'image_path': f'scene_{scene_idx:04d}_rgb.png',
            'location': list(cam_obj.location),
            'rotation': list(cam_obj.rotation_euler),
            'focal_length': cam_obj.data.lens,
            'intrinsics': {'fx': 1500, 'fy': 1500, 'cx': 512, 'cy': 384}
        }
        with open(os.path.join(camera_info_dir, f'scene_{scene_idx:04d}_camera_{i}_info.json'), 'w') as f:
            json.dump(cam_meta, f, indent=2)
