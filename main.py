import os
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def load_dataset(dataset_path):
    X, y = [], []

    for letter in os.listdir(dataset_path):
        letter_path = os.path.join(dataset_path, letter)
        if not os.path.isdir(letter_path):
            continue

        for image_name in os.listdir(letter_path):
            image_path = os.path.join(letter_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize the image to a fixed size (e.g., 100x100)
                image = cv2.resize(image, (100, 100))
                X.append(image.flatten())
                y.append(letter)

    return np.array(X), np.array(y)

def train_classifier(dataset_path):
    # Load the dataset
    X, y = load_dataset(dataset_path)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier (Support Vector Machine)
    classifier = SVC(kernel='linear')

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    accuracy = classifier.score(X_test, y_test)
    print(f"Classifier accuracy: {accuracy:.2f}")

    # Save the classifier to a file
    joblib.dump(classifier, 'classifier.pkl')

if __name__ == "__main__":
    # Replace 'path/to/your/dataset' with the path to the ASL alphabet dataset
    dataset_path = 'path/to/your/dataset'
    train_classifier(dataset_path)
