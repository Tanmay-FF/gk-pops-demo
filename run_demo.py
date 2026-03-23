#!/usr/bin/env python3
import subprocess
import sys
import os
import platform

def run_command(command, shell=False):
    """Run a system command and stream output."""
    process = subprocess.Popen(command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        sys.exit(f"Command failed: {command}")

def main():
    # Step 1: Create virtual environment
    print("POPS demonstration:")
    print("Step 1: Creating virtual environment...")
    venv_dir = "venv_" + os.path.basename(os.getcwd()).lower()
    if not os.path.exists(venv_dir):
        run_command([sys.executable, "create_virtual_env.py"])
    else:
        print("Virtual environment folder already exists... activating now")

    # Step 2: Run the demo
    print("Step 2: Run demo...")

    if platform.system() == "Windows":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    run_command([venv_python, "app_poc_v2.py"])
    print("Please wait for it to load on your browser...")

if __name__ == "__main__":
    main()