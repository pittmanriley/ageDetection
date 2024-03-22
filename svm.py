"""
This file implements a support vector machine (SVM) in order to compare its results
with the results from model.py. I couldn't get this to work, so it's not included in 
my final report. 
"""


from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np


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

def evaluate(predictions, y_test):
    # Calculate accuracy within a range of 10 units
    correct_predictions = sum(1 for pred, actual in zip(predictions, y_test) if abs(pred - actual) <= 10)
    total_samples = len(y_test)
    accuracy_within_range = correct_predictions / total_samples
    print("Accuracy within 10 units:", accuracy_within_range)
    

def main():
    input_folder = 'starter_training_set'
    target_size = (224, 224)

    # Load and preprocess the dataset
    images, labels = load_dataset(input_folder, target_size)
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # instantiate SVM classifier 
    svm_classifier = svm.SVC(kernel='linear')

    # Train SVM classifier
    svm_classifier.fit(X_train, y_train)

    # Make predictions on test data
    predictions = svm_classifier.predict(X_test)
    evaluate(predictions, y_test)


if __name__ == '__main__': 
    main()