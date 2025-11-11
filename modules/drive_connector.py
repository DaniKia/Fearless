"""
Google Drive connector for accessing files in both Replit and Colab environments.
"""

import os
import sys
import io
from pathlib import Path

def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def is_replit():
    """Check if running in Replit."""
    return 'REPL_ID' in os.environ

def setup_drive_access():
    """
    Set up Google Drive access based on the environment.
    Returns True if successful, False otherwise.
    """
    if is_colab():
        print("Detected Colab environment")
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            print("Google Drive mounted successfully")
            return True
        except Exception as e:
            print(f"Error mounting Google Drive in Colab: {e}")
            return False
    
    elif is_replit():
        print("Detected Replit environment")
        print("Using Replit's Google Drive integration")
        return True
    
    else:
        print("Unknown environment - Drive access may not work")
        return False

def get_drive_service():
    """
    Get Google Drive API service for Replit environment.
    Returns None if in Colab (uses mounted filesystem instead).
    """
    if is_colab():
        return None
    
    if is_replit():
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            import json
            
            creds_json = os.environ.get('GOOGLE_DRIVE_CREDENTIALS')
            if not creds_json:
                print("Warning: GOOGLE_DRIVE_CREDENTIALS not found in environment")
                return None
            
            creds_data = json.loads(creds_json)
            creds = Credentials.from_authorized_user_info(creds_data)
            service = build('drive', 'v3', credentials=creds)
            return service
        except Exception as e:
            print(f"Error setting up Drive API service: {e}")
            return None
    
    return None

def list_files_in_folder(folder_path):
    """
    List files in a Google Drive folder.
    Works in both Colab (using mounted filesystem) and Replit (using Drive API).
    
    Args:
        folder_path: Path to folder in Drive
        
    Returns:
        List of file paths
    """
    if is_colab():
        if os.path.exists(folder_path):
            return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        else:
            print(f"Folder not found: {folder_path}")
            return []
    
    elif is_replit():
        service = get_drive_service()
        if not service:
            print("Drive service not available. Make sure Google Drive integration is connected.")
            return []
        
        try:
            folder_name = os.path.basename(folder_path)
            results = service.files().list(
                q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                fields="files(id, name)"
            ).execute()
            
            folders = results.get('files', [])
            if not folders:
                print(f"Folder '{folder_name}' not found in Drive")
                return []
            
            folder_id = folders[0]['id']
            file_results = service.files().list(
                q=f"'{folder_id}' in parents and mimeType!='application/vnd.google-apps.folder'",
                fields="files(id, name, mimeType)"
            ).execute()
            
            files = file_results.get('files', [])
            return [f['name'] for f in files]
        
        except Exception as e:
            print(f"Error listing files: {e}")
            return []
    
    return []

def download_file(file_path, local_path):
    """
    Download a file from Google Drive to local storage.
    In Colab, just copies from mounted drive. In Replit, uses Drive API.
    
    Args:
        file_path: Path to file in Drive
        local_path: Where to save the file locally
        
    Returns:
        True if successful, False otherwise
    """
    if is_colab():
        if os.path.exists(file_path):
            import shutil
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy(file_path, local_path)
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    
    elif is_replit():
        service = get_drive_service()
        if not service:
            return False
        
        try:
            from googleapiclient.http import MediaIoBaseDownload
            
            file_name = os.path.basename(file_path)
            results = service.files().list(
                q=f"name='{file_name}'",
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            if not files:
                print(f"File '{file_name}' not found")
                return False
            
            file_id = files[0]['id']
            request = service.files().get_media(fileId=file_id)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            fh = io.FileIO(local_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            return True
        
        except Exception as e:
            print(f"Error downloading file: {e}")
            return False
    
    return False
