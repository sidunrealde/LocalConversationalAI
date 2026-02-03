#!/usr/bin/env python3
"""
Pre-flight checks for Local Conversational AI Agent
Validates Python packages, Ollama connectivity, CUDA availability, and disk space.
Run this BEFORE starting the Streamlit app.
"""

import sys
import os
import subprocess
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
CHECKMARK = "✓"
CROSS = "✗"
ARROW = "→"


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{BLUE}{'='*60}")
    print(f"{title}")
    print(f"{'='*60}{RESET}")


def check_pass(message):
    """Print a passing check."""
    print(f"{GREEN}{CHECKMARK}{RESET} {message}")


def check_fail(message):
    """Print a failing check."""
    print(f"{RED}{CROSS}{RESET} {message}")


def check_warn(message):
    """Print a warning."""
    print(f"{YELLOW}⚠{RESET}  {message}")


def check_info(message):
    """Print info message."""
    print(f"{BLUE}ℹ{RESET}  {message}")


# ============================================================================
# CHECKS
# ============================================================================

def check_python_version():
    """Check if Python version is 3.10+"""
    print_section("1. Python Version")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        check_pass(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        check_fail(f"Python {version.major}.{version.minor} (require 3.10+)")
        return False


def check_packages():
    """Check if required packages are installed."""
    print_section("2. Required Packages")
    
    required = [
        "streamlit",
        "ollama",
        "piper",
        "faster_whisper",
        "soundfile",
        "numpy",
        "PIL",
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            check_pass(f"{pkg}")
        except ImportError:
            check_fail(f"{pkg} (missing)")
            missing.append(pkg)
    
    if missing:
        check_warn(f"Missing packages: {', '.join(missing)}")
        check_info(f"Run: pip install -r requirements.txt")
        return False
    return True


def check_ollama():
    """Check if Ollama server is running."""
    print_section("3. Ollama Server")
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        if result.returncode == 0:
            check_pass("Ollama server is running")
            
            # Parse models
            lines = result.stdout.strip().split("\n")
            if len(lines) > 1:  # More than just header
                check_info(f"Available models:")
                for line in lines[1:]:
                    if line.strip():
                        model_name = line.split()[0]
                        check_info(f"  {ARROW} {model_name}")
            else:
                check_warn("No models downloaded. Run: ollama pull qwen2.5:7b")
            return True
        else:
            check_fail("Ollama server error")
            return False
    except FileNotFoundError:
        check_fail("Ollama not installed or not in PATH")
        check_info("Download from: https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        check_fail("Ollama server not responding (timeout)")
        return False
    except Exception as e:
        check_fail(f"Error checking Ollama: {e}")
        return False


def check_cuda():
    """Check if CUDA is available."""
    print_section("4. GPU Support (CUDA)")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            check_pass(f"CUDA available ({device_count} device(s))")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                total_memory = device_props.total_memory / 1e9
                check_info(f"  Device {i}: {device_name} ({total_memory:.1f} GB)")
            return True
        else:
            check_warn("CUDA not available (CPU-only mode, will be slow)")
            return True  # Not required, just slower
    except ImportError:
        check_warn("PyTorch not found (CUDA check skipped)")
        return True


def check_disk_space():
    """Check available disk space."""
    print_section("5. Disk Space")
    
    space_needed = {
        "Whisper models": 3,
        "TTS voices": 1,
        "Ollama models": 5,
        "Cache & data": 2,
        "Buffer": 2,
    }
    
    total_needed = sum(space_needed.values())
    
    try:
        import shutil
        stat = shutil.disk_usage(".")
        available_gb = stat.free / (1024**3)
        
        check_info(f"Space requirements:")
        for item, gb in space_needed.items():
            check_info(f"  {ARROW} {item}: {gb} GB")
        
        check_info(f"Total needed: {total_needed} GB")
        check_info(f"Available: {available_gb:.1f} GB")
        
        if available_gb >= total_needed:
            check_pass(f"Sufficient disk space ({available_gb:.1f} GB available)")
            return True
        else:
            check_fail(f"Insufficient disk space (need {total_needed} GB, have {available_gb:.1f} GB)")
            return False
    except Exception as e:
        check_warn(f"Could not check disk space: {e}")
        return True


def check_directories():
    """Check if required directories exist or can be created."""
    print_section("6. Project Structure")
    
    required_dirs = [
        "modules",
        "data",
        "data/voices",
        "data/avatars",
        "logs",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            check_pass(f"{dir_path}/")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                check_pass(f"{dir_path}/ (created)")
            except Exception as e:
                check_fail(f"{dir_path}/ ({e})")
                all_ok = False
    
    return all_ok


def check_files():
    """Check if required files exist."""
    print_section("7. Required Files")
    
    required_files = [
        "chat.py",
        "requirements.txt",
        "README.md",
        "SETUP.md",
        "TESTING.md",
        "modules/__init__.py",
        "modules/logger.py",
        "modules/tts.py",
        "modules/asr.py",
        "modules/ollama_client.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            check_pass(file_path)
        else:
            check_fail(f"{file_path} (missing)")
            all_ok = False
    
    return all_ok


def summary(checks):
    """Print summary and recommendation."""
    print_section("Summary")
    
    passed = sum(checks.values())
    total = len(checks)
    
    print(f"\n{passed}/{total} checks passed\n")
    
    if all(checks.values()):
        print(f"{GREEN}✓ All systems ready!{RESET}\n")
        print("You can now run: streamlit run chat.py")
        return 0
    else:
        failed = [name for name, result in checks.items() if not result]
        print(f"{RED}✗ {len(failed)} check(s) failed:{RESET}\n")
        for name in failed:
            print(f"  {ARROW} {name}")
        print(f"\nSee messages above for how to fix.")
        return 1


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all checks."""
    print(f"\n{BLUE}{'='*60}")
    print("Local Conversational AI Agent - Pre-flight Checks")
    print(f"{'='*60}{RESET}\n")
    
    checks = {
        "Python Version": check_python_version(),
        "Required Packages": check_packages(),
        "Ollama Server": check_ollama(),
        "CUDA/GPU": check_cuda(),
        "Disk Space": check_disk_space(),
        "Directories": check_directories(),
        "Files": check_files(),
    }
    
    exit_code = summary(checks)
    
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
