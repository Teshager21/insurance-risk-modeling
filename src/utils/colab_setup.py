import os
import subprocess
import sys


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
    service_account_path: str = "secrets/dvc-drive-remote-8f00f1ce2758.json",
    dvc_remote_name: str = "my_gdrive_remote",
):
    """
    Automate project setup for Google Colab with support for GDrive DVC remote.

    Steps:
    - Clone GitHub repository
    - Install Python requirements
    - Install DVC and configure GDrive remote with service account
    - Pull DVC-tracked data
    - Install the project in editable mode
    - Add src/ folder to Python path for imports

    Args:
        repo_url (str): GitHub repository URL.
        repo_name (str): Directory name to clone into.
        requirements_file (str): Pip requirements file.
        dvc_remote_type (str): DVC remote type (e.g., "gdrive").
        service_account_path (str): Path to GDrive service account JSON file.
        dvc_remote_name (str): Name of the configured DVC remote.
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

    # Add src folder to sys.path for imports
    src_path = os.path.abspath("src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    print(f"🐍 Added '{src_path}' to sys.path")

    print(f"📦 Installing dependencies from {requirements_file} ...")
    subprocess.run(["pip", "install", "-r", requirements_file], check=True)

    print(f"📦 Installing DVC with remote '{dvc_remote_type}' support ...")
    subprocess.run(["pip", "install", f"dvc[{dvc_remote_type}]"], check=True)

    if dvc_remote_type == "gdrive" and os.path.exists(service_account_path):
        print("🔐 Configuring DVC to use GDrive service account ...")
        subprocess.run(
            [
                "dvc",
                "remote",
                "modify",
                dvc_remote_name,
                "gdrive_service_account_json_file_path",
                service_account_path,
            ],
            check=True,
        )
    else:
        print("⚠️ Warning: Service account JSON not found. GDrive remote may fail.")

    print("📡 Pulling DVC-tracked data ...")
    subprocess.run(["dvc", "pull", "-r", dvc_remote_name], check=True)

    print("🔧 Installing project package in editable mode ...")
    subprocess.run(["pip", "install", "-e", "."], check=True)

    print("✅ Google Colab setup completed successfully!")
