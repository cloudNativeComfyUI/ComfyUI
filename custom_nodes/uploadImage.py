import torch
import io
import os
import sys
import hashlib
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
import folder_paths


import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class LoadImageCloud:
    account_url = "https://teststoragej4bw9l.blob.core.windows.net"
    default_credential = DefaultAzureCredential()
    container_name = "images"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        file_names = cls.list_blob_names()
        return {"required": {"file_name": (file_names,)}}

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    @classmethod
    def list_blob_names(cls):
        blob_service_client = BlobServiceClient(cls.account_url, credential=cls.default_credential)
        container_client = blob_service_client.get_container_client(cls.container_name)

        return [blob.name for blob in container_client.list_blobs()]

    def load_image(self, file_name):
        # Download the blob to a local file
        # Add 'DOWNLOAD' before the .txt extension so you can see both files in the data directory
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(self.account_url, credential=self.default_credential)
        image_bytes_stream = self.download_to_byte_stream(blob_service_client, file_name)

        print(f"Downloaded blob from Azure")
        img = Image.open(image_bytes_stream)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    
    def download_blob_to_file(self, blob_service_client: BlobServiceClient, file_name: str):
        print(f"Attemping blob dl {file_name}")
        downloaded_file_path = os.path.join(r'input', file_name)
        blob_client = blob_service_client.get_blob_client(container="images", blob=file_name)
        print(f"Succeeded fetch image from AZ")
        with open(file=downloaded_file_path, mode="wb") as sample_blob:
            download_stream = blob_client.download_blob()
            sample_blob.write(download_stream.readall())
        return downloaded_file_path

    def download_to_byte_stream(self, blob_service_client: BlobServiceClient, file_name: str):
        print(f"Attemping blob dl {file_name}")
        blob_client = blob_service_client.get_blob_client(container="images", blob=file_name)
        print(f"Succeeded fetch image from AZ")

        download_stream = blob_client.download_blob()
        bytes_stream = io.BytesIO(b'')
        download_stream.readinto(bytes_stream)

        return bytes_stream

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    # @classmethod
    # def VALIDATE_INPUTS(s, image):
    #     if not folder_paths.exists_annotated_filepath(image):
    #         return "Invalid image file: {}".format(image)

    #     return True
    
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadImageCloud": LoadImageCloud
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageCloud": "Load Image Cloud"
}
