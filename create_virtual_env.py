import os
import subprocess
import sys
import zipfile
from pathlib import Path

def delete_empty_folders(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Deleted empty folder: {dirpath}")
            except OSError as e:
                print(f"Failed to delete {dirpath}: {e}")

def unzip_and_delete(folder_path = os.getcwd()):
    zip_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(folder_path, filename)
            extract_dir = os.path.join(folder_path, os.path.splitext(filename)[0])
            zip_files.append(zip_path)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                print(f"Unzipped: {filename}")
            except zipfile.BadZipFile:
                print(f"Bad zip file: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    for zip_file in zip_files:
        try:
            os.remove(zip_file)
            print(f"Deleted: {zip_file}")
        except FileNotFoundError:
            print(f"{zip_file} not found...")
        except Exception as e:
            print(f"Error with {zip_file}: e")

def install_cuda_if_missing():
    """
    Attempts to install CUDA 12.2 if nvcc is not found.
    """
    try:
        subprocess.check_output(["nvcc", "--version"])
        print("CUDA already installed.")
        return
    except Exception:
        print("CUDA not found. Installing CUDA 12.2...")

    if os.name == "nt":
        # Windows installer
        url = "https://developer.nvidia.com/compute/cuda/12.2.0/network_installers/cuda_12.2.0_windows_network.exe"
        installer = "cuda_installer.exe"

        subprocess.check_call(["powershell", "-Command", f"Invoke-WebRequest {url} -OutFile {installer}"])
        subprocess.check_call([installer, "-s"])  # silent install

    else:
        # Linux (Ubuntu example)
        subprocess.check_call([
            "bash", "-c",
            "wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run -O cuda.run"
        ])
        subprocess.check_call(["chmod", "+x", "cuda.run"])
        subprocess.check_call(["sudo", "./cuda.run", "--silent", "--toolkit"])

def add_cuda_path_linux(venv_path, cuda_bin_path="/usr/local/cuda-12.2/bin"):
    """
    Adds CUDA bin folder to PATH inside the venv bin/activate script for Linux/Mac.
    """
    activate_file = venv_path / "bin" / "activate"

    if not activate_file.exists():
        print(f"No activate script found at {activate_file}, skipping CUDA PATH addition.")
        return

    export_line = f'export PATH={cuda_bin_path}:$PATH\n'

    with open(activate_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if any(cuda_bin_path in line for line in lines):
        print("CUDA path already exists in activate script, skipping.")
        return

    # Insert at the end or near the top
    lines.append(export_line)

    backup_path = activate_file.with_suffix(".backup")
    if not backup_path.exists():
        activate_file.rename(backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    with open(activate_file, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Check that the path is added correctly
    nvcc_path = Path(cuda_bin_path) / "nvcc"
    if not nvcc_path.exists():
        print(f"Warning: {nvcc_path} not found. CUDA may not be correctly installed.")


    print(f"Added CUDA bin path to {activate_file}")

def add_cuda_path_windows(venv_path, 
                          cuda_bin_path=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"):
    """
    Adds CUDA path to Windows activate.bat.
    """
    activate_bat = venv_path / "Scripts" / "activate.bat"
    if not activate_bat.exists():
        print(f"No activate.bat found at {activate_bat}, skipping CUDA PATH addition.")
        return

    cuda_line = f'set PATH={cuda_bin_path};%PATH%\n'

    with open(activate_bat, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if any(cuda_bin_path in line for line in lines):
        print("CUDA path already exists in activate.bat, skipping.")
        return

    insert_index = 1
    for i, line in enumerate(lines):
        if line.strip():
            insert_index = i + 1
            break

    lines.insert(insert_index, cuda_line)
    nvcc_path = Path(cuda_bin_path) / "nvcc.exe"

    # Check that the path is added correctly
    if not nvcc_path.exists():
        print(f"Warning: {nvcc_path} not found. CUDA may not be correctly installed.")

    backup_path = activate_bat.with_suffix(".bat.backup")
    if not backup_path.exists():
        activate_bat.rename(backup_path)
        print(f"Backup of activate.bat created at {backup_path}")
    else:
        print(f"Backup activate.bat already exists at {backup_path}")

    with open(activate_bat, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Added CUDA bin path to {activate_bat}")

def setup_venv(venv_dir='venv', requirements_file='requirements.txt'):
    venv_path = Path(venv_dir)
    req_path = Path(requirements_file)

    print(f"Creating venv: {venv_path}")
    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file '{requirements_file}' not found.")

    # Step 1: Create venv 
    subprocess.check_call([sys.executable, '-m', 'venv', str(venv_path)])
    print(f"Virtual environment created at: {venv_path}")

    # Step 2: Add CUDA path BEFORE installing packages
    print("\nAdding CUDA to PATH...")
    if os.name == 'nt':
        add_cuda_path_windows(venv_path)
        python_path = venv_path / 'Scripts' / 'python.exe'
        pip_path = venv_path / 'Scripts' / 'pip.exe'
        cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"
    else:
        add_cuda_path_linux(venv_path)
        python_path = venv_path / 'bin' / 'python'
        pip_path = venv_path / 'bin' / 'pip'
        cuda_bin_path = "/usr/local/cuda-12.2/bin"

    # Step 3: Upgrade pip
    subprocess.check_call([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'])

    # Step 4: Install all other packages...
    env = os.environ.copy()
    env["PATH"] = f"{cuda_bin_path}{os.pathsep}{env['PATH']}"

    print("\nInstalling packages from requirements.txt..")
    subprocess.check_call([str(pip_path), 'install', '-r', str(req_path)], env=env)
    print(f"Installed packages from '{requirements_file}'")

    # Step 5: Explicitly install onnxruntime-gpu from official CUDA index
    print("\nInstalling onnxruntime-gpu from ONNX's official CUDA wheel source...")
    subprocess.check_call([
        str(pip_path),
        'install',
#        'onnxruntime-gpu==1.17.0',
#        '--extra-index-url',
#        'https://download.onnxruntime.ai/onnxruntime_stable_cu118.html'
          'onnxruntime-gpu'
    ], env=env)

if __name__ == "__main__":
    venv_name = "venv_" + Path(os.getcwd()).stem.lower()
    requirements = 'requirements.txt'

    try:
        delete_empty_folders(".")
        unzip_and_delete()
        setup_venv(venv_name, requirements)
        print(f"\n::venv_name::{venv_name}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)