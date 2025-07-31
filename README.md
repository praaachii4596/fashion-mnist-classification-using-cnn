# Fashion MNIST Image Classification with CNN (Google Colab)

This project demonstrates a complete workflow for classifying fashion images using a Convolutional Neural Network (CNN) in TensorFlow/Keras. The model is trained and evaluated on the Fashion MNIST dataset.

---

## Overview

- **Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - 60,000 grayscale 28x28 images of clothing (10 categories)
- **Model:** Convolutional Neural Network (CNN) with two Conv2D+MaxPooling2D stacks and dense layers.
- **Platform used:** Google Colab.

---

## Features

- Loads and preprocesses Fashion MNIST data.
- Visualizes sample training images with correct labels.
- Builds a CNN suitable for image recognition.
- Trains with real-time validation monitoring.
- Evaluates model on test split.
- Predicts and displays classification for new/unseen images.
- Plots accuracy and loss curves for both train and validation data.

---

## How to Use

### 1. Open or Upload Notebook

- Open in [Google Colab](https://colab.research.google.com/) or upload this notebook to your Drive.

### 2. Import the Dependencies

- All required libraries (`tensorflow`, `numpy`, `matplotlib`) are pre-installed in Colab.

### 3. Run Each Section Step by Step

#### Data Loading and Preprocessing
- Loads Fashion MNIST.
- Reshapes data for CNN input `(samples, 28, 28, 1)`.
- Normalizes pixel values to [0,1].

#### Class Labels
- Maps numeric labels (0-9) to human-readable clothing categories.

#### Visualization
- Shows 10 sample images with their true class names.

#### Model Architecture
- Sequential CNN:
    - `Conv2D(32) → MaxPooling2D`  
    - `Conv2D(64) → MaxPooling2D`
    - `Flatten → Dense(128) → Dense(10, softmax)`

#### Compilation
- Optimizer: Adam
- Loss Function: Sparse categorical cross-entropy

#### Training
- Trains for 5 epochs with 10% of data held out for validation.

#### Evaluation
- Reports model accuracy and loss on test set.

#### Prediction & Display
- Predicts the category of any test image and shows true vs. predicted label.

#### Training Progress Visualization
- Plots loss and accuracy curves for training and validation, helping to see progress and evaluate possible overfitting.

---

## Model Performance

- Typically reaches ~90%+ test accuracy after several epochs.
- Training and validation curves provide insights into learning and signs of overfitting.
