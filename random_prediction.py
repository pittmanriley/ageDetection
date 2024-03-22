"""
This file is used to create a random age prediction for each image in the 
dataset. It generates a random prediction for each age and evaluates the
accuracy. 
"""

import os 
import random

def calculate_accuracy(predicted, actual_ages): 
    correct_predictions = sum(1 for pred, actual in zip(predicted, actual_ages) if abs(pred - actual) <= 10)
    total_samples = len(actual_ages)
    accuracy_within_range = correct_predictions / total_samples
    print("Accuracy within 10 units:", accuracy_within_range)
    
    
def main(): 
    # Replace 'input_folder' with the path to your image folder
    input_folder = 'utkcropped'

    # Create a list to store the predicted ages
    predicted_ages = []
    actual_ages = []

    # Iterate through each image in the folder
    for filename in os.listdir(input_folder): 
        split = filename.split('_')
        actual_ages.append(int(split[0]))
        
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # generate a random age between 1 and 116
            predicted_age = random.randint(1, 116)
            predicted_ages.append(predicted_age)
    
    calculate_accuracy(predicted_ages, actual_ages)
    
        

if __name__ == '__main__':
    main()