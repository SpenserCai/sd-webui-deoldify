import os
import launch
from modules import paths_internal
import urllib.request
from tqdm import tqdm
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

# Define model URLs
MODEL_URLS = {
    "stable_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth",
    "artistic_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeArtistic_gen.pth",
    "video_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth"
}

DEPS = [
    'wandb',
    'fastai==1.0.60',
    'tensorboardX',
    'ffmpeg',
    'ffmpeg-python',
    'yt-dlp',
    'opencv-python',
    'Pillow'
]

models_dir = os.path.join(paths_internal.models_path, "deoldify")

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

def download(url, path):
    """Download a file from a URL to a specific path, showing progress."""
    if os.path.exists(path):
        return False  # File already exists, no need to download
    try:
        with urllib.request.urlopen(url) as request, open(path, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(path)}",
            total=int(request.headers.get('Content-Length', 0)),
            unit='B',
            unit_scale=True,
            unit_divisor=1024
        ) as progress:
            for chunk in iter(lambda: request.read(4096), b""):
                f.write(chunk)
                progress.update(len(chunk))
        return True
    except Exception as e:
        print(f"Failed to download {os.path.basename(path)}: {e}")
        return False

def download_models():
    """Download all models if they don't exist locally."""
    models_already_downloaded = True
    for name, url in MODEL_URLS.items():
        path = os.path.join(models_dir, os.path.basename(url))
        if download(url, path):
            models_already_downloaded = False
    if models_already_downloaded:
        print("All models for DeOldify are already downloaded.")

def check_and_install_dependencies():
    """Install required dependencies for DeOldify, with version checks."""
    dependencies_met = True
    for dep in DEPS:
        package_name, *version = dep.split('==')
        version = version[0] if version else None
        if not check_package_installed(package_name, version):
            dependencies_met = False
            print(f"Installing {dep} for DeOldify extension.")
            launch.run_pip(f"install {dep}", package_name)
    if dependencies_met:
        print("All requirements for the DeOldify extension are already installed.")

def check_package_installed(package_name, version=None):
    """Check if a package is installed with an optional version."""
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name
    try:
        pkg_resources.require(package_spec)
        return True
    except (DistributionNotFound, VersionConflict):
        return False

if __name__ == "__main__":
    download_models()
    check_and_install_dependencies()
