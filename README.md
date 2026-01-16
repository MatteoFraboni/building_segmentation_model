# Building Object Segmentation with Deep Learning (Docker-based)

This repository contains a deep learning pipeline for **object segmentation of buildings** using a pre-trained model.<br>
The model provided is trained using **High-resolution orthophotos** so for the prediction you must provide images with the same quality.<br>
The project is fully containerized using **Docker** to ensure reproducibility and ease of setup.

The repository provides:
- A complete preprocessing–training–prediction pipeline
- A **pre-trained model (10 epochs)** ready for inference created with U_net architecture
- Scripts to retrain the model if desired (optional)
- Clear step-by-step instructions to run everything

---

## 1. Project Overview

The pipeline is composed of **four main scripts**, each responsible for a specific step:

1. **image_preprocessing**  
   Preprocesses the original dataset by splitting images and masks into patches for model training.

2. **model_training**  
   Defines the deep learning model, performs training, and saves the trained weights.<br>
 **NOTE: the model is trained using High-resolution orthophotos so is not suited for satellitary images** <br>
**you must retrain the model using satellitary images if you want to use those as predictions**

3. **image_pred_preprocessing**  
   Preprocesses a user-selected test image by splitting it into patches (images only) for inference.

4. **model_prediction**  
   Loads the trained model, performs segmentation on the image patches, and reconstructs the final full-size mask.

---

## 2. Requirements

### Software Requirements

- **Docker** (mandatory)
- Git (optional, for cloning the repository)

### Hardware Requirements (Recommended)

- GPU with CUDA support (recommended for training)
- CPU-only execution is supported but slower

---

## 3. Installing Docker

If Docker is not already installed, follow the official instructions:

- Docker installation guide:  
  https://docs.docker.com/get-docker/

After installation, verify Docker is working:

```bash
docker --version
```

---

## 4. Repository Setup

### 4.1 Clone the Repository

```bash
git  clone  https://github.com/MatteoFraboni/building_segmentation_model.git 
cd  building_segmentation_model
```

### 4.2 Dataset Download

Due to size limitations, the dataset used for training is **not included** in this repository.

You can download the dataset from the following link:

```
https://board.unimib.it/datasets/9kbc6zdn7b/2
```

After downloading, extract the dataset to a local directory, for example:

```
/path/to/dataset/
```

The dataset is expected to contain:
- Original images
- Corresponding segmentation masks

---

## 5. Docker Image Build

The project uses a Docker image defined by a `Dockerfile` included in the repository. <br>
This DockerFile is already equipped with **TensorFlow 2.15.0** preinstalled so there is no need for installation.

### 5.1 Build the Docker Image

From the repository root directory:

```bash
docker build -t building-segmentation .
```

### 5.2 Run the Docker Container

```bash
docker  run  -it building-segmentation .
```
---

## 6. Path Configuration

Inside the scripts, some file paths are defined explicitly.

You should replace them with placeholders such as:

- `PATH_TO_ORIGINAL_IMAGES`
- `PATH_TO_ORIGINAL_MASKS`
- `PATH_TO_PATCHES_IMAGES`
- `PATH_TO_PATCHES_MASKS`
- `PATH_TO_SAVED_MODEL`
- `PATH_TO_TEST_IMAGE`

**Make sure to update these paths according to your local directory structure before running the scripts.**

---

## 7. Step-by-Step Pipeline

### Step 1 (Optional): Image Preprocessing for Training

**Script:** `image_preprocessing.py`

This step prepares the dataset for training by:
- Loading original images and masks
- Splitting them into smaller patches
- Saving image patches and mask patches separately

Run this step **only if you want to retrain the model**.

```bash
python image_preprocessing.py
```

---

### Step 2 (Optional): Model Training

**Script:** `model_training.py`

This script:
- Builds the deep learning segmentation model
- Trains it on the preprocessed patches
- Saves the trained model to disk

A **pre-trained model (10 epochs)** is already provided, so this step can be skipped for inference-only usage.

```bash
python model_training.py
```

The trained model will be saved to:

```
PATH_TO_SAVED_MODEL
```

---

### Step 3: Preprocessing for Prediction

**Script:** `image_pred_preprocessing.py`

This step:
- Loads a single test image selected by the user
- Splits the image into patches
- Saves the patches for inference

```bash
python image_pred_preprocessing.py
```

Make sure the path to the test image is correctly set:

```
PATH_TO_TEST_IMAGE
```

---

### Step 4: Model Prediction and Mask Reconstruction

**Script:** `model_prediction.py`

This script:
- Loads the trained (or pre-trained) model
- Performs segmentation on each image patch
- Reassembles the predicted patches
- Outputs a full-size segmentation mask

```bash
python model_prediction.py
```

The final output is a **single reconstructed segmentation mask** corresponding to the input image.

---

## 8. Output

The final results include:
- Predicted segmentation masks for buildings
- Intermediate patch-level predictions (optional)

All outputs are saved to the directories specified in the path configuration.

---

## 9. Notes

- Training is optional; inference can be performed directly using the provided pre-trained model.
- Make sure all paths are correctly configured before execution.
- Docker ensures consistent execution across different systems.

---

For questions or issues, please open an issue on GitHub.
