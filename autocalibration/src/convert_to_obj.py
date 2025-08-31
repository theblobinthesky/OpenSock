import bpy
import sys
import os

# Addons are enabled by default in Blender

# Parse arguments
if '--' in sys.argv:
    args = sys.argv[sys.argv.index('--') + 1:]
    if len(args) < 2:
        print("Usage: blender -b -P convert_to_obj.py -- <input_file> <output_dir>")
        sys.exit(1)
    input_file = args[0]
    output_dir = args[1]
else:
    print("No arguments provided")
    sys.exit(1)

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Import based on file extension
ext = os.path.splitext(input_file)[1].lower()
if ext == '.fbx':
    bpy.ops.import_scene.fbx(filepath=input_file)
elif ext == '.dae':
    bpy.ops.import_scene.collada(filepath=input_file)
elif ext == '.blend':
    bpy.ops.wm.open_mainfile(filepath=input_file)
else:
    print(f"Unsupported file type: {ext}")
    sys.exit(1)

# Get the base name
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_obj = os.path.join(output_dir, f"{base_name}.obj")

# Export to OBJ
bpy.ops.wm.obj_export(
    filepath=output_obj,
    export_materials=True,
    path_mode='RELATIVE'
)

print(f"Converted {input_file} to {output_obj}")