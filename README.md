# Cat vs. Dog Classification Using Convolutional Neural Network (CNN)

This repository contains the implementation of a CNN model for classifying images of cats and dogs. The model is built using TensorFlow and Keras and is designed to distinguish between the two categories using a sample dataset from Kaggle.

## Project Overview

The goal of this project is to develop a deep learning model that can automatically classify images of cats and dogs. This is a binary classification problem where the model is trained to predict whether an image is a cat or a dog. The model leverages a Convolutional Neural Network (CNN), a powerful deep learning architecture commonly used in image classification tasks.

## Repository Contents

- **dogs_vs_cats.ipynb**: Jupyter notebook containing the complete implementation of the CNN model, including data preprocessing, model building, training, evaluation, and prediction visualization.
- **README.md**: This file, containing an overview and details on how to run the project.
- **data/**: Directory to store the training and validation data for cats and dogs.
- **models/**: Directory to save the trained CNN models.

## Tools and Technologies

- **TensorFlow**: A popular deep learning framework used to build the CNN model.
- **Keras**: A high-level API that simplifies model creation.
- **Matplotlib**: Used for data visualization and plotting training performance graphs.
- **Python**: The core programming language used to implement the project.

## CNN Architecture Overview

- **Input Layer**: The input images are resized to 150x150 pixels with 3 channels (RGB).
- **Convolutional Layers**: We use multiple Conv2D layers to extract features from the input images.
- **Pooling Layers**: MaxPooling2D is used to reduce the dimensions of the feature maps while retaining important features.
- **Fully Connected Layers**: Dense layers are added for classification after the flattened output from convolutional layers.
- **Dropout Layer**: Dropout is used to prevent overfitting by randomly dropping units during training.
- **Output Layer**: A single neuron with a sigmoid activation function for binary classification (cat or dog).

## Features

- **Data Augmentation**: Techniques such as rotation, flipping, and zooming are used to increase the variability of the training data and reduce overfitting.
- **Transfer Learning**: VGG16 pre-trained on ImageNet is used for fine-tuning the model to improve classification performance.
- **Evaluation Metrics**: Model evaluation is performed using accuracy, precision, recall, and a confusion matrix to assess performance.

## How to Run the Project

1. **Clone the Repository**
    ```bash
    git clone https://https://github.com/subhashpolisetti/cat-dog-classification-cnn
    cd cat-dog-classification-cnn
    ```

2. **Install Dependencies**
    Ensure you have Python 3.x and the following libraries installed:
    ```bash
    pip install tensorflow matplotlib numpy opencv-python
    ```

3. **Download the Dataset**
    Download the Kaggle Dogs vs. Cats dataset and place the files in the `data/` directory. Alternatively, you can use a sample dataset of your choice and update the code accordingly.

4. **Run the Jupyter Notebook**
    Run the `dogs_vs_cats.ipynb` notebook to preprocess the data, build the model, and train it. You can visualize the training process and evaluate the model using the validation set.

## How to Run the Project
  Reference Links:
    For a comprehensive step-by-step guide, including dataset exploration and code execution, please refer to the complete walkthrough video available on YouTube.

**Youtube vedio**: [YouTube Video Explanation Link](https://www.youtube.com/playlist?list=PL6O21IOHvBmf4VAAySH9Kmu2DJgm9XOeN)

**Medium Article :**: [Medium Article Link](https://medium.com/@subhashr161347/classifying-dogs-and-cats-using-a-convolutional-neural-network-cnn-e58718ebf2e8)

**ChatGPT Chat Transcript :**: [ChatGPT Chat Transcript Link](https://chatgpt.com/share/67036fc0-ddb0-8009-9f9f-5c466cc07146)

**Colab Implementation File :**: [Colab Implementation File Link](https://colab.research.google.com/drive/1hDnyJKnLlQQFpzhDeYgCFgW8qm2rlen-?usp=sharing)

    


