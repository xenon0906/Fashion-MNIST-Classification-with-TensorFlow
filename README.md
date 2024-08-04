# Fashion MNIST Classification with TensorFlow

## Project Overview

This project aims to build a machine learning model using TensorFlow to classify images from the Fashion MNIST dataset. The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28x28 pixels), as seen in the figure below.

## Dataset

The Fashion MNIST dataset contains:

- 60,000 training images
- 10,000 test images

Each image is a 28x28 pixel grayscale image, associated with a label from 10 classes:

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

## Project Steps

### 1. Importing Libraries

We start by importing the necessary libraries:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

### 2. Loading the Dataset

Load the Fashion MNIST dataset from TensorFlow's Keras API:

```python
fmnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
```

### 3. Exploring the Data

Check the shape and content of the data:

```python
index = 0
np.set_printoptions(linewidth=320)
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')
plt.imshow(training_images[index], cmap='Greys')
```

### 4. Normalizing the Data

Normalize the images to have values between 0 and 1:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

### 5. Building the Model

Define the neural network model:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 6. Compiling the Model

Compile the model with an optimizer, loss function, and metrics:

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### 7. Training the Model

Train the model with the training data:

```python
model.fit(training_images, training_labels, epochs=10)
```

### 8. Evaluating the Model

Evaluate the model using the test data:

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
```

### 9. Making Predictions

Use the trained model to make predictions:

```python
predictions = model.predict(test_images)
```

## Results

The model is evaluated on the test set, achieving an accuracy of approximately X% (replace X with the actual accuracy achieved).

## Conclusion

This project demonstrates how to build and train a neural network model to classify images from the Fashion MNIST dataset using TensorFlow. The model can be further improved by experimenting with different architectures, optimizers, and hyperparameters.

## Requirements

- TensorFlow 
- NumPy
- Matplotlib

## How to Run

1. Clone this repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the Jupyter notebook to execute the code and train the model.

## Future Work

- Experiment with different neural network architectures.
- Implement data augmentation to improve model robustness.
- Explore transfer learning with pre-trained models.
