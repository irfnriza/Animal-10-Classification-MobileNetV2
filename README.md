# Animal-10 Classification with MobileNetV2

This repository contains a deep learning project for classifying 10 different animal species using the MobileNetV2 architecture. The project demonstrates data exploration, preprocessing, model training, evaluation, and conversion to various deployment formats.

## Table of Contents

- [Project Description](#project-description)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Class Labels](#class-labels)
- [Installation](#installation)
- [Usage](#usage)
- [Exported Models](#exported-models)

## Project Description

The goal of this project is to build an image classification model that can accurately identify 10 distinct animal categories. It utilizes transfer learning with a pre-trained MobileNetV2 model and fine-tunes it with additional layers for this specific task. The project is designed to be end-to-end, covering data handling, model development, and deployment readiness.

## Features

- **Exploratory Data Analysis (EDA):** Scripts to understand dataset distribution and image characteristics.
- **Efficient Data Preprocessing:** Includes dataset splitting into training, validation, and test sets, and efficient data generation with augmentation using `ImageDataGenerator`.
- **Transfer Learning with MobileNetV2:** Utilizes the MobileNetV2 architecture pre-trained on ImageNet as a backbone.
- **Custom Classification Head:** Adds custom `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `GlobalAveragePooling2D`, `Dense`, and `Dropout` layers for classification.
- **Model Training and Evaluation:** Functions for training the model with callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau) and evaluating its performance.
- **Multi-format Model Export:** Converts the trained model to:
    - TensorFlow SavedModel
    - TensorFlow Lite (TFLite)
    - TensorFlow.js
- **Inference Functions:** Provides utilities to perform predictions using the exported Keras/SavedModel and TFLite models.

## Dataset

The project uses the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10) from Kaggle Hub. This dataset contains images of 10 different animal species. The notebook includes code to download and structure this dataset for training.

## Model Architecture

The core of the model is built using `MobileNetV2` without its top (fully connected) layers. This base model's weights are typically frozen initially, and then a custom classification head is appended. The custom layers include:

- `Conv2D` layers with `ReLU` activation and `BatchNormalization`.
- `MaxPooling2D` for downsampling.
- `GlobalAveragePooling2D` to reduce spatial dimensions.
- `Dense` layers with `ReLU` and `Dropout` for regularization.
- A final `Dense` layer with `softmax` activation for 10-class classification.

## Class Labels

The model is trained to classify the following 10 animal species:
- `cane` (dog)
- `cavallo` (horse)
- `elefante` (elephant)
- `farfalla` (butterfly)
- `gallina` (chicken)
- `gatto` (cat)
- `mucca` (cow)
- `pecora` (sheep)
- `ragno` (spider)
- `scoiattolo` (squirrel)

## Installation

To set up the project locally, you'll need Python and the necessary libraries. It is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd animal-10-classification-mobilenetv2
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    The required libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary workflow for this project is demonstrated in the `notebook.ipynb` file. Follow the steps within the notebook:

1.  **Download Dataset:** The notebook automatically downloads the `animals10` dataset from Kaggle Hub.
2.  **Run EDA:** Explore the dataset's structure, image resolutions, and class distribution.
3.  **Preprocess Data:** The `split_dataset` function prepares the data for training, validation, and testing. `create_efficient_data_generators` sets up data augmentation and efficient loading.
4.  **Build and Train Model:** The `build_mobilenetv2_sequential_model` function constructs the model, and `train_model` handles the training process.
5.  **Evaluate Model:** Use `evaluate_model` and `evaluate_and_print_metrics` to assess the model's performance with a classification report and confusion matrix.
6.  **Export Model:** The `export_model_to_all_formats` function saves the trained model in SavedModel, TFLite, and TensorFlow.js formats.
7.  **Inference:** Use the `infer_image` function to test predictions on new images with different model formats.

## Exported Models

The project exports the trained model into the following formats for various deployment scenarios:

- **SavedModel:** Located at `saved_model`
- **TFLite:** Located at `tflite/model.tflite`
- **TensorFlow.js:** Located at `tfjs_model`

Additionally, a `model_metadata.json` file is generated, providing key information about the model, its input shape, number of classes, preprocessing steps, and paths to the exported formats.
