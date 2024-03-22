"""
This file is used for croppping the images in the starter training set. The program
uses the cv2 library to detect faces in the images, and then the image is cropped around
that face. The new dataset is saved as full_cropped_faces.

Note: These images were used for some of the tests, but some of the images ended up 
not being useful because it failed to crop an image around the face of the individual. 
"""

import cv2
import os

def detect_and_crop_faces(image_folder, output_folder):
    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iterate over each image in the input folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full path to the image
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)

            # change the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces 
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Iterate over each detected face in the image
            for i, (x, y, w, h) in enumerate(faces):
                # Crop the image
                cropped_face = image[y:y+h, x:x+w]
                cropped_face = cv2.resize(cropped_face, (224, 224))  # Example: resize to 100x100

                # Save the cropped face 
                output_path = os.path.join(output_folder, f"{filename}_cropped_face_{i}.jpg")
                cv2.imwrite(output_path, cropped_face)


def main():
    input_folder = 'starter_training_set'  # Path to the input image
    output_folder = 'full_cropped_faces'  # Path to the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    detect_and_crop_faces(input_folder, output_folder)
    
    
if __name__ == '__main__': 
    main()