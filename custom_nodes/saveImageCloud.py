import torch
import io
import os
import sys
import json

from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision

import comfy.model_management
from comfy.cli_args import args


import folder_paths
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings

class SaveImageCloud:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"images": ("IMAGE", ),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        print(f"save_images >>> images type: {type(images)}")
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for image in images:
            file = f"{filename}_{counter:05}_.png"
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Convert Pillow Image to Bytes
            image_byte_array = io.BytesIO()
            img.save(image_byte_array, format='PNG')
            img_bytes = image_byte_array.getvalue()
            print(f"save_images >>> img_bytes type: {type(img_bytes)}")

            # Upload to Azure Blob Storage   
            self.upload_blob_from_bytes("images", f"{filename}_{counter:05}_.png", img_bytes)
            counter += 1

        return { "ui": { "images": results } }

    def upload_blob_from_bytes(self, container_name: str, filename: str, blob_bytes):
        # Connect to AZ Blob Storage
        account_url = "https://teststoragej4bw9l.blob.core.windows.net"
        default_credential = DefaultAzureCredential()

         # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        container_client = blob_service_client.get_container_client(container=container_name)
        blob_client = container_client.get_blob_client(filename)

        # Upload to Azure Blob Storage
        content_settings = ContentSettings(content_type="image/png")
        blob_client.upload_blob(blob_bytes, content_settings=content_settings, overwrite=True)

    def upload_blob_tags(self, container_name: str, filename: str):
        account_url = "https://teststoragej4bw9l.blob.core.windows.net"
        default_credential = DefaultAzureCredential()
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        container_client = blob_service_client.get_container_client(container=container_name)
        with open(f'./output/{filename}', "rb") as data:
            container_client.upload_blob(name=filename, data=data, overwrite=True)
        os.remove(f'./output/{filename}')
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SaveImageCloud": SaveImageCloud
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageCloud": "Save Image Cloud"
}
