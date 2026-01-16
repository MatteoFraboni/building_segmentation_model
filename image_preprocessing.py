# importing the libraries 

import cv2
import numpy as np
import os


def extract_patches(image, mask, patch_size=256, stride=256):
    img_patches = []
    mask_patches = []
    
    # h and w correspond to height and width of the images
    h, w = image.shape[:2]

    # making the height tiling from 0 to the last tile using the stride as step (from top to bottom)
    for y in range(0, h - patch_size + 1, stride):
        # # making the width tiling from 0 to the last tile using the stride as step (from left to right)
        for x in range(0, w - patch_size + 1, stride):

            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            img_patches.append(img_patch)
            mask_patches.append(mask_patch)

    return img_patches, mask_patches


# loading directories for images and masks and defining final patches size
PATCH_SIZE = 256 # final patch size
STRIDE = 256  # final patch size
IMAGES_DIR  = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/images"
MASKS_DIR  = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/masks"
PATCH_IMG_DIR = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/patches/images"
PATCH_MASK_DIR = "/workspace/progetto_dati_satellitari/object_detection/buildings/data/patches/masks"

# creating name for final patches 
patch_id = 0

# grabbing one image at a time for ram saving
for filename in os.listdir(IMAGES_DIR):
    img_path = os.path.join(IMAGES_DIR, filename)
    mask_path = os.path.join(MASKS_DIR, filename)

    # loading selected image and converting to BGR for OpenCV
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # loading selected mask 
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # creating patches for selected image and mask
    img_patches, mask_patches = extract_patches(image, mask)

    # for each patch extracted saving the image to path
    for img_p, mask_p in zip(img_patches, mask_patches):
        # saving current image patch to disk
        img_name = f"img_{patch_id}.png"
        img_save_path = os.path.join(PATCH_IMG_DIR, img_name)

        cv2.imwrite(img_save_path, cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR))
        
        # saving current mask patch to disk
        mask_name = f"mask_{patch_id}.png"
        mask_save_path = os.path.join(PATCH_MASK_DIR, mask_name)

        cv2.imwrite(mask_save_path, mask_p)
        print(f"\rSaved image and mask patch: {patch_id}", end="", flush=True)
        patch_id += 1


