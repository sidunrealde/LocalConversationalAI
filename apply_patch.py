import os

ctPath = r"C:\Users\SiddarthaGonnabattul\AppData\Local\Programs\Python\Python313\Lib\site-packages\ctranslate2\__init__.py"

# Read the file
with open(ctPath, 'r') as f:
    content = f.read()

# Replace the problematic lines
patched = content.replace(
    'add_dll_directory(f"{package_dir}/../_rocm_sdk_core/bin")',
    'rocm_core = os.path.join(package_dir, "../_rocm_sdk_core/bin")\n        if os.path.exists(rocm_core):\n            add_dll_directory(rocm_core)'
).replace(
    'add_dll_directory(f"{package_dir}/../_rocm_sdk_libraries_custom/bin")',
    'rocm_lib = os.path.join(package_dir, "../_rocm_sdk_libraries_custom/bin")\n        if os.path.exists(rocm_lib):\n            add_dll_directory(rocm_lib)'
)

# Write back
with open(ctPath, 'w') as f:
    f.write(patched)

print("Patched successfully!")
