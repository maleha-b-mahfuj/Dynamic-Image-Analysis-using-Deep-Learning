# PyTorch Image Classification Project

**Developed By:** Md Ishraq Tanzim

## Overview
This project demonstrates the implementation of Convolutional Neural Network (CNN) and a simpler neural network model for image classification using PyTorch. It includes data preprocessing, model training, evaluation, and visualization techniques to effectively classify images and analyze model performance. The primary objective is to implement a machine learning model that can classify hand-written digit images taken from the MNIST dataset. The challenge, solved in this project, involves using a limited dataset to learn a classifier that generalizes well to unseen test data. The dataset includes images where the top sub-image acts as a pointer to either the middle or bottom sub-images, which determines the label of the whole image based on the digit displayed in the top sub-image.

## Features
- CNN and simple neural network models for image classification.
- Data preprocessing for PyTorch compatibility.
- Model training and evaluation using accuracy metrics.
- Visualization of image data and classification results.

## Dependencies
- PyTorch
- Matplotlib
- Pandas
- scikit-learn

## Repository Contents
- `ImageClassifier.py` - Main script for training and evaluating the image classification model.
- `TrainingDataset.csv` - Dataset file used for training the model.
- `ValidationDataset.csv` - Dataset file used for validating the model's performance.
- `requirements.txt` - Lists all the Python dependencies required for this project.


## Getting Started
To get started, clone this repository and install the required dependencies:
```
git clone https://github.com/Ishraq3002/Dynamic-Image-Analysis-using-Deep-Learning
pip install -r requirements.txt
```

## Usage
To train and evaluate the model, run:
```
python ImageClassifier.py
```
Ensure you have the training and validation datasets in the project directory before running the script.

## Contributing
Contributions to improve the models or extend the project's capabilities are welcome. Please submit a pull request or open an issue for discussion.

## Contact
- Name: Md Ishraq Tanzim
- Email: mdishraqtanzim@cmail.carleton.ca

