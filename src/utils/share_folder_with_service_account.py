from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Path to your service account key JSON
import os

credentials_path = os.path.expanduser(
    "~/.config/gdrive/dvc-drive-remote-8f00f1ce2758.json"
)

SERVICE_ACCOUNT_FILE = credentials_path

# Folder ID of your Google Drive folder you want to share
FOLDER_ID = "1fPOrXKds0PNYQbKNbSDtC5j51MXm9gY7"

# The service account email you want to share the folder with
SERVICE_ACCOUNT_EMAIL = "dvc-remote@dvc-drive-remote.iam.gserviceaccount.com"

# Scopes required for Drive API
SCOPES = ["https://www.googleapis.com/auth/drive"]


def share_folder_with_service_account():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build("drive", "v3", credentials=credentials)

    permission_body = {
        "type": "user",
        "role": "writer",  # or 'reader' if you want read-only access
        "emailAddress": SERVICE_ACCOUNT_EMAIL,
    }

    try:
        response = (
            service.permissions()
            .create(fileId=FOLDER_ID, body=permission_body, fields="id")
            .execute()
        )

        print(f"Permission ID: {response.get('id')}")
        print(f"Folder shared with {SERVICE_ACCOUNT_EMAIL} successfully!")

    except HttpError as error:
        print(f"An error occurred: {error}")


if __name__ == "__main__":
    share_folder_with_service_account()
