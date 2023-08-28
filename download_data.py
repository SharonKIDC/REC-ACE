import requests
import zipfile
import os
import argparse


def download_file_from_google_drive(id):
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    return response


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def download_and_unzip(url: str, destination_folder):
    
    if url.startswith('https://'):
        # Make a request to download the zip file
        response = requests.get(url)
    else:
        response = download_file_from_google_drive(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Create the destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)

        # Save the zip file to the destination folder
        zip_file_path = os.path.join(destination_folder, 'downloaded_file.zip')
        print('Downloading Data')
        with open(zip_file_path, 'wb') as zip_file:
            zip_file.write(response.content)
        print('Downloading Finished')

        print('Unzipping Data')
        # Unzip the downloaded file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        print('Unzziping Finished')

        # Remove the downloaded zip file
        print('Removing zip file')
        os.remove(zip_file_path)
        print("Download and unzip successful.")

    else:
        print("Failed to download the file.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download and unzip a zip file.')
    parser.add_argument('-url',
                        type=str,
                        help='URL of the zip file to download or a Google Drive file ID',
                        default='https://storage.googleapis.com/gresearch/red-ace/data.zip')
    parser.add_argument('-dest_path',
                        type=str,
                        help='Destination folder path for the extracted files',
                        default='./')
    args = parser.parse_args()

    download_and_unzip(args.url, args.dest_path)

