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
        
        # Check if Drive is already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("Google Drive already mounted")
            return True
        
        # Try to mount if not already mounted
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            print("Google Drive mounted successfully")
            return True
        except Exception as e:
            print(f"Error mounting Google Drive in Colab: {e}")
            print("Please make sure you've mounted Drive in your Colab notebook first:")
            print("  from google.colab import drive")
            print("  drive.mount('/content/drive')")
            return False
    
    elif is_replit():
        print("Detected Replit environment")
        print("Using Replit's Google Drive integration")
        return True
    
    else:
        print("Unknown environment - Drive access may not work")
        return False

def get_access_token():
    """
    Get access token from Replit Google Drive connection.
    Returns access token string or None if unavailable.
    """
    try:
        import requests
        
        hostname = os.environ.get('REPLIT_CONNECTORS_HOSTNAME')
        x_replit_token = None
        
        if os.environ.get('REPL_IDENTITY'):
            x_replit_token = 'repl ' + os.environ['REPL_IDENTITY']
        elif os.environ.get('WEB_REPL_RENEWAL'):
            x_replit_token = 'depl ' + os.environ['WEB_REPL_RENEWAL']
        
        if not hostname or not x_replit_token:
            print("Warning: Replit connector environment variables not found")
            return None
        
        url = f'https://{hostname}/api/v2/connection?include_secrets=true&connector_names=google-drive'
        headers = {
            'Accept': 'application/json',
            'X_REPLIT_TOKEN': x_replit_token
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        items = data.get('items', [])
        
        if not items:
            print("Warning: No Google Drive connection found")
            return None
        
        connection_settings = items[0]
        settings = connection_settings.get('settings', {})
        
        access_token = settings.get('access_token') or settings.get('oauth', {}).get('credentials', {}).get('access_token')
        
        if not access_token:
            print("Warning: No access token found in connection settings")
            return None
        
        return access_token
        
    except Exception as e:
        print(f"Error getting access token: {e}")
        return None

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
            
            access_token = get_access_token()
            if not access_token:
                print("Warning: Could not get access token from Replit connection")
                print("Please make sure Google Drive is connected in your Replit project")
                return None
            
            creds = Credentials(token=access_token)
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
