import os
import zipfile
import subprocess

BASE_DIR = './data'
os.makedirs(BASE_DIR, exist_ok=True)
PROC_DIR = os.path.join(BASE_DIR, 'hoan_processed')
os.makedirs(PROC_DIR, exist_ok=True)
RAW_DIR = os.path.join(BASE_DIR, 'hoan_raw_pdb')
os.makedirs(RAW_DIR, exist_ok=True)


ZENODO_URL = "https://zenodo.org/record/10291581/files/Nomburg_2023_structures.zip?download=1"

def download_pdbs_fast():
    zip_path = os.path.join(RAW_DIR, "virome_pdbs.zip")

    is_valid_zip = False
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as archive:
                print("Existing zip file found. Verifying integrity...")
                is_valid_zip = True
        except zipfile.BadZipFile:
            print("Existing zip file is corrupted. Deleting and re-downloading.")
            os.remove(zip_path)
        except Exception as e:
            print(f"An unexpected error occurred while checking zip file: {e}. Deleting and re-downloading.")
            os.remove(zip_path)

    if not is_valid_zip:
        print("Downloading dataset using aria2 (multiple connections)...")
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # Utilisation de subprocess.run pour un appel propre sans interférence de Shell
        cmd = [
            "aria2c", 
            "-x", "16", 
            "-s", "16", 
            "-d", RAW_DIR, 
            "-o", "virome_pdbs.zip", 
            ZENODO_URL
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Aria2 encountered an error during download: {e}")
        except FileNotFoundError:
            print("ERROR: aria2c command not found. Ensure it is installed and in your PATH.")

        # Vérification après téléchargement
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as archive:
                    print("Download successful and zip file is valid.")
            except zipfile.BadZipFile:
                print("ERROR: Downloaded file is corrupted. Please check the URL or try again.")
                os.remove(zip_path)
        else:
            print("ERROR: Download did not complete successfully. Zip file not found.")
    else:
        print("Zip file already exists and is valid.")

download_pdbs_fast()