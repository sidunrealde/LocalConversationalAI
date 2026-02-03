"""
Patch for ctranslate2 ROCm issue on Windows.
This must be imported before any module that uses faster_whisper.
"""

import sys
import os

if sys.platform == "win32":
    # Patch ctranslate2 before it's imported
    import ctypes
    import glob
    
    # Get the ctranslate2 module directory
    try:
        from importlib.resources import files
        package_dir = str(files("ctranslate2"))
    except ImportError:
        import pkg_resources
        package_dir = pkg_resources.resource_filename("ctranslate2", "")
    
    # Add main directory
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(package_dir)
        except Exception:
            pass
        
        # Safely add ROCm directories if they exist
        for subdir in ["_rocm_sdk_core/bin", "_rocm_sdk_libraries_custom/bin"]:
            rocm_path = os.path.join(package_dir, "..", subdir)
            if os.path.exists(rocm_path):
                try:
                    add_dll_directory(rocm_path)
                except Exception:
                    pass
        
        # Load DLLs
        for library in glob.glob(os.path.join(package_dir, "*.dll")):
            try:
                ctypes.CDLL(library)
            except Exception:
                pass
