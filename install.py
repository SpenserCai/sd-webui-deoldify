import os
import subprocess
import urllib.request
from tqdm import tqdm
from modules import paths_internal

# Define model URLs
MODEL_URLS = {
    "stable_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth",
    "artistic_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeArtistic_gen.pth",
    "video_model": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth"
}

# Clean dependency list — compatible with Python 3.10 and modern SD envs
DEPS = [
    "fastai==1.0.60",
    "tensorboardX",
    "ffmpeg-python",
    "yt-dlp",
    "opencv-python",
    "Pillow>=9.0.0,<10.0.0"
]

models_dir = os.path.join(paths_internal.models_path, "deoldify")
os.makedirs(models_dir, exist_ok=True)


def download(url, path):
    """Download a file from a URL to a specific path with progress bar."""
    if os.path.exists(path):
        return False
    try:
        with urllib.request.urlopen(url) as request, open(path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(path)}",
            total=int(request.headers.get("Content-Length", 0)),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in iter(lambda: request.read(4096), b""):
                f.write(chunk)
                progress.update(len(chunk))
        return True
    except Exception as e:
        print(f"Failed to download {os.path.basename(path)}: {e}")
        return False


def download_models():
    """Ensure all model weights are present."""
    all_downloaded = True
    for name, url in MODEL_URLS.items():
        path = os.path.join(models_dir, os.path.basename(url))
        if download(url, path):
            all_downloaded = False
    if all_downloaded:
        print("All models for DeOldify are already downloaded.")


def install_dependencies():
    """Install required pip packages safely, ignoring conflicts."""
    print("\nInstalling DeOldify dependencies...\n")
    for dep in DEPS:
        print(f"Installing {dep} ...")
        cmd = [
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "--prefer-binary",
            "--upgrade",
            "--quiet",
            dep,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {dep}, continuing anyway.")

    print("\nDeOldify dependency setup complete.\n")


if __name__ == "__main__":
    download_models()
    install_dependencies()
