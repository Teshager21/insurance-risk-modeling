# File: src/utils/colab_setup.py

import os
import subprocess


def in_colab() -> bool:
    """Detect if the code is running inside a Google Colab environment."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def bootstrap_colab(
    repo_url: str,
    repo_name: str,
    requirements_file: str = "requirements.txt",
    dvc_remote_type: str = "gdrive",
):
    """
    Automate project setup for Google Colab.

    Steps performed:
    - Clone GitHub repository (if not present)
    - Install Python dependencies
    - Install DVC with remote support
    - Pull DVC-tracked data
    - Install the project package in editable mode

    Args:
        repo_url (str): HTTPS URL of the GitHub repository to clone.
        repo_name (str): Folder name for the cloned repository.
        requirements_file (str): Path to requirements.txt.
        dvc_remote_type (str): DVC remote type (e.g., 'gdrive', 's3').
    """
    if not in_colab():
        print("🖥️ Not running in Google Colab. Skipping setup.")
        return

    print("🚀 Starting Google Colab environment setup...")

    if not os.path.exists(repo_name):
        print(f"📥 Cloning repository from {repo_url} ...")
        subprocess.run(["git", "clone", repo_url], check=True)
    else:
        print(f"✅ Repository '{repo_name}' already exists.")

    os.chdir(repo_name)

    print(f"📦 Installing dependencies from {requirements_file} ...")
    subprocess.run(["pip", "install", "-r", requirements_file], check=True)

    print(f"📦 Installing DVC with remote '{dvc_remote_type}' support ...")
    subprocess.run(["pip", "install", f"dvc[{dvc_remote_type}]"], check=True)

    print("📡 Pulling DVC-tracked data ...")
    subprocess.run(["dvc", "pull"], check=True)

    print("🔧 Installing project package in editable mode ...")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    print("✅ Google Colab setup completed successfully!")
