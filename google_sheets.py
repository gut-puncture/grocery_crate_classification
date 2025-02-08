# google_sheets.py
import os
import io
import time
import logging
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

logging.basicConfig(level=logging.INFO)

credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if not credentials_path or not os.path.exists(credentials_path):
    credentials_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_json:
        temp_path = "service_account_temp.json"
        with open(temp_path, "w") as f:
            f.write(credentials_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path

SCOPES = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]

# Initialize credentials
creds = service_account.Credentials.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=SCOPES
)


# Initialize Google Drive service
drive_service = build('drive', 'v3', credentials=creds)

# Initialize gspread client and open the sheet
gc = gspread.authorize(creds)
SHEET_ID = os.environ.get("GOOGLE_SHEET_ID")
if not SHEET_ID:
    logging.error("GOOGLE_SHEET_ID environment variable not set.")
    raise Exception("GOOGLE_SHEET_ID not set.")
sheet = gc.open_by_key(SHEET_ID).sheet1

# (Optional) Use a specific Drive folder if provided
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID")

def upload_image_to_drive(image_bytes: bytes, filename: str) -> str:
    """
    Uploads an image (JPEG) to Google Drive and returns its public URL.
    """
    file_metadata = {
        'name': filename,
        'mimeType': 'image/jpeg'
    }
    if DRIVE_FOLDER_ID:
        file_metadata['parents'] = [DRIVE_FOLDER_ID]
    media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype='image/jpeg')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')
    
    # Make the file publicly accessible
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    drive_service.permissions().create(fileId=file_id, body=permission).execute()
    
    # Generate and return the public URL (this URL works with the IMAGE() function in Sheets)
    file_url = f"https://drive.google.com/uc?id={file_id}"
    logging.info(f"Image uploaded to Drive with URL: {file_url}")
    return file_url

def append_to_sheet(image_url: str, threshold: str, classification: str):
    """
    Append a new row to the Google Sheet with:
      - Crate_Image: The URL of the uploaded image.
      - Threshold_type: The quality type ("premium" or "average").
      - Classification: The Gemini classification result.
    """
    row = [image_url, threshold, classification]
    sheet.append_row(row)
    logging.info("Row appended to Google Sheet: " + str(row))

def update_google_sheet(image_bytes: bytes, threshold: str, classification: str):
    """
    Upload the image to Google Drive and then update the Google Sheet with the
    image URL, threshold, and classification.
    """
    filename = f"crate_{int(time.time())}.jpg"
    image_url = upload_image_to_drive(image_bytes, filename)
    append_to_sheet(image_url, threshold, classification)
