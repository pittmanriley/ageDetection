"""
This file is the main code that loads the data, processes the images from the dataset, 
sets up the model, predicts the ages, and evaluates those predictions.
"""

import cv2
import numpy as np
import os 
import tensorflow as tf
from keras import layers, models
import random
from sklearn.model_selection import train_test_split
from keras.constraints import non_neg

def preprocess(image_path, target_size):
    # Read the image
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)

    # Normalize pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0
    return normalized_image

# Load and preprocess the entire dataset
def load_dataset(folder_path, target_size):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            age_label = int(filename.split('_')[0])  
            processed_image = preprocess(image_path, target_size)
            images.append(processed_image)
            labels.append(age_label)

    img_array = np.array(images)
    label_array = np.array(labels)
    return img_array, label_array

def evaluate(model, predictions, y_test, X_test):
    # Calculate accuracy within a range of 10 units
    correct_predictions = sum(1 for pred, actual in zip(predictions, y_test) if abs(pred - actual) <= 10)
    total_samples = len(y_test)
    accuracy_within_range = correct_predictions / total_samples
    print("Accuracy within 10 units:", accuracy_within_range)
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test MAE (Mean Absolute Error):", test_mae)
    
    
def record_results(output_folder, predictions, y_test):
    with open(output_folder, 'w') as f:
        # Iterate over the indices of one of the lists (assuming both lists have the same length)
        for i in range(len(predictions)):
            # Write the index and elements from both lists separated by a comma
            f.write(f"{predictions[i]},{y_test[i]}\n")
    

def main(): 
    input_folder = 'starter_training_set'
    output_folder = 'results_utk_cropped.txt'
    target_size = (224, 224)

    # Load and preprocess the dataset
    images, labels = load_dataset(input_folder, target_size)
    print('done with load dataset')

    # Split data into training and testing sets. 0.2 means it will reserve 20% of the data set for testing
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print('done with train test split')

    # Define a simple CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='relu', kernel_constraint=non_neg())  # Output layer for age prediction
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    print('done with compile')

    # Train the model
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)
    print('done with model fit')
    
    # Evaluate the model
    predictions = model.predict(X_test)
    
    # record_results(output_folder, predictions, y_test)
    evaluate(model, predictions, y_test, X_test)



if __name__ == '__main__': 
    main()
    

####### RESULTS: ########

# with the starter_training_set: 
# first test with 10 epochs: 0.41074950690335305, loss was 509.47 and MAE was 16.1
# second test was with 15 epochs: 0.46844181459566075, loss was 

# with the full_cropped_faces: 
# first test with 15 epochs: 0.5463502015226153, loss was 341.3414 and MAE was 12.9148
# second test with 30 epochs: 0.5620241827138379, loss was 367.8287 and MAE was 12.689

# with the utkcropped: 
# first test with 20 epochs: 0.7056094474905104, loss was 117.4864 and MAE was 7.8896
# second test with 35 epochs: 0.7068747363981442, loss was 116.9774 and MAE was 7.890

# with utkcropped and accuracy within 2 units: 
# done with 7 epochs: 0.16364403205398567, loss was 146.9920654296875 and MAE was 9.135

