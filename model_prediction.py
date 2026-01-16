# importing the libraries 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def predict_patch(model, img_path):
    # loading the single patch and converting colors
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resizing and adding a third layer for keras
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    pred = model.predict(img)
    pred = np.squeeze(pred)

    # the mask is created where the model predict values below the treshold
    mask = (pred > THRESHOLD).astype(np.uint8)
    return mask

STRIDE = 256
PATCH_SIZE = 256
THRESHOLD = 0.5
BASE_PATH = "PATH_TO_PREDICTION_DIRECTORY" # this directory must be composed of 3 folders: one named "original" containing the original test image, one named "patches" containing the patches created with image_pred_preprocessing.py, and one named "detection" to store the predicted mask
PATCHES_PATH = os.path.join(BASE_PATH, "patches")
ORIGINAL_PATH = os.path.join(BASE_PATH, "original")
OUTPUT_PATH = os.path.join(BASE_PATH, "detection")

# name of the test image and model
img_name = "NAME_OF_THE_TEST_IMAGE"
model_name = "Trained_model_10_epochs"


# loading model
model = tf.keras.models.load_model(f"PATH_TO_SAVED_MODEL/{model_name}", compile=False) 


# reading original image
original_img = cv2.imread(os.path.join(ORIGINAL_PATH, img_name))
output_img = original_img.copy()
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

H, W = original_img.shape[:2]

# initializing final mask dimensions from original image
final_mask = np.zeros((H, W), dtype=np.uint8)


# building the final mask
patch_id = 0

for y in range(0, H - PATCH_SIZE + 1, STRIDE):
    for x in range(0, W - PATCH_SIZE + 1, STRIDE):

        patch_name = f"img_{patch_id}.png"
        patch_path = os.path.join(PATCHES_PATH, patch_name)

        mask_patch = predict_patch(model, patch_path)

        final_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = mask_patch

        patch_id += 1


# saving final mask
cv2.imwrite(
    os.path.join(OUTPUT_PATH, f"mask_{img_name}_final.png"),
    final_mask * 255
)

# finding contours for final bounding box
contours, _ = cv2.findContours(
    final_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# drawing final image with bounding boxes

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if w * h < 100:  
        continue

    cv2.rectangle(
        output_img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imwrite(
    os.path.join(OUTPUT_PATH, f"{img_name}_detection.png"),
    output_img
)



