"""
Module with the definition of classes and functions for storing and loading data from storage.
"""

import os

from typing import Optional
from google.cloud import storage
from pathlib import Path


class StorageManager:
    """
    Class with methods for extracting and loading objects to Storage

    Args:
    -----
        project: str
            GCP project to which the storage for storage belongs
        bucket: str
            Name of the storage where the files will be stored
    """

    def __init__(self, gcp_project: str, bucket_name: Optional[str] = None) -> None:
        self.project = gcp_project
        self.client = storage.Client(project=gcp_project)
        if bucket_name is not None:
            self.bucket = self.client.get_bucket(bucket_name)

    def upload_file(self, local_file: str, gcs_file: str):
        """
        Method that uploads a file to the bucket

        Args:
        -----
            local_file (str): 
                name (with its path) of the file to upload.
            bucket_file (str): 
                name (with its path) of the resulting object in the GCS bucket.
        """
        blob = self.bucket.blob(gcs_file)

        try:
            blob.upload_from_filename(local_file)
        except Exception as e:
            raise Exception(f"Error uploading file {local_file}: {e}")

    def upload_dir(self, local_dir: str, gcs_dir: str):
        """
        Method that uploads the contents of a directory to the bucket

        Args:
        -----
            local_dir (str): 
                name (with its path) of the directory to upload.
            gcs_dir (str):
                name (with its path) of the resulting directory in the GCS bucket.
        """
        local_files = os.listdir(local_dir)

        for local_file in local_files:
            remote_path = f'{gcs_dir}/{local_file}'
            blob = self.bucket.blob(remote_path)
            local_file_path = os.path.join(local_dir, local_file)
            try:
                blob.upload_from_filename(local_file_path)
            except Exception as e:
                raise Exception(f"Error uploading file {local_file}: {e}")

    def download_file(self, local_file: str, gcs_file: str):
        """
        Method that downloads a file from the bucket in gcs

        Args:
        -----
            file (str):
                name (with its path) of the file to download.
            file_name (str):
                name (with its path) of the file in the GCS bucket.
        """
        blob = self.bucket.blob(gcs_file)

        try:
            blob.download_to_filename(local_file)
        except Exception as e:
            raise Exception(f"Error downloading file {gcs_file}: {e}")

    def download_dir(self, local_dir: str, gcs_dir: str):
        """
        Method that loads the contents of a directory to the bucket

        Args:
        -----
            local_dir (str): 
                name (with its path) of the directory to download.
            gcs_dir (str):
                name (with its path) of the directory in the GCS bucket.
        """

        blobs = self.bucket.list_blobs(prefix=gcs_dir)
        for blob in blobs:
            file_name = blob.name.split("/")[-1]
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            local_file = os.path.join(local_dir, file_name)
            try:
                blob.download_to_filename(local_file)
            except Exception as e:
                raise Exception(f"Error downloading file {file_name}: {e}")

    def upload_file_to_uri(self, local_file: str, gcs_uri: str):
        """
        Method that uploads a file to gcs at a specific uri

        Args:
        -----
            loca_file (str):
                name (with its local path) of the file to upload.
            gcs_uri (str):
                uri of the file in the storage to save the file.
        """
        blob = storage.blob.Blob.from_string(gcs_uri, client=self.client)
        try:
            blob.upload_from_filename(local_file)
        except Exception as e:
            raise Exception(f"Error uploading file {local_file}: {e}")

    def upload_dir_to_uri(self, local_dir: str, gcs_uri: str):
        """
        Method that uploads the contents of a directory to a directory in storage
        with a specific uri.

        Args:
        -----
            local_dir (str):
                name (with its path) of the file to upload.
            gcs_uri (str):
                uri of the directory in the storage to save the directory.
        """
        local_files = os.listdir(local_dir)

        for local_file in local_files:
            remote_path = f'{gcs_uri}/{local_file}'
            blob = storage.blob.Blob.from_string(remote_path, client=self.client)
            local_file_path = os.path.join(local_dir, local_file)
            try:
                blob.upload_from_filename(local_file_path)
            except Exception as e:
                raise Exception(f"Error uploading file {local_file}: {e}")

    def download_file_from_uri(
        self,
        gcs_uri: str,
        local_file: str,
    ):
        """
        Method that downloads a file from the bucket in gcs

        Args:
        -----
            gcs_uri (str):
                uri in the storage to download the file.
            local_file (str):
                name (with its path) of the file to download.

        """
        blob = storage.blob.Blob.from_string(gcs_uri, client=self.client)

        try:
            blob.download_to_filename(local_file)
        except Exception as e:
            raise Exception(f"Error downloading file {gcs_uri}: {e}")

    def download_dir_from_uri(
        self,
        gcs_uri: str,
        local_dir: str,
    ):
        """
        Method that loads the contents of a directory to the bucket

        Args:
        -----
            gcs_uri (str):
            local_dir (str):
                name (with its path) of the directory to download.
        """

        # Remove 'gs://' prefix from gcs_uri and get the bucket name and prefix
        bucket_name, prefix = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = self.client.bucket(bucket_name)

        # List blobs with the given prefix
        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            # Get the relative path of the file within the GCS directory
            relative_path = blob.name[len(prefix) :].lstrip("/")

            # Create the full local path
            local_file = os.path.join(local_dir, relative_path)

            # Create necessary directories
            Path(os.path.dirname(local_file)).mkdir(parents=True, exist_ok=True)

            try:
                # Download the file
                blob.download_to_filename(local_file)
            except Exception as e:
                raise Exception(f"Error downloading file {blob.name}: {e}")

        print("All files have been downloaded.")

    def delete_file(self, gcs_file: str):
        """
        Method that deletes a file from the GCS bucket

        Args:
        -----
            gcs_file (str):
                Name (with its path) of the file in the bucket to delete.
                It must have the format 'folder/file_name' and be relative to the bucket.
        """
        blob = self.bucket.blob(gcs_file)

        try:
            blob.delete()
        except Exception as e:
            raise Exception(f"Error deleting file {gcs_file}: {e}")

    def delete_dir(self, gcs_dir: str):
        """
        Method that deletes a directory from the GCS bucket

        Args:
        -----
            gcs_dir (str):
                name (with its path) of the directory in the GCS bucket.
        """

        blobs = self.bucket.list_blobs(prefix=gcs_dir)
        for blob in blobs:
            try:
                blob.delete()
            except Exception as e:
                raise Exception(f"Error deleting file {blob.name}: {e}")