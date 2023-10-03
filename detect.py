import cv2
import mediapipe as mp
import joblib

# Initialize Mediapipe's Hand class
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def load_classifier(model_path):
    # Load the pre-trained classifier
    classifier = joblib.load(model_path)
    return classifier

def predict_hand_sign(classifier, landmarks):
    # Predict the hand sign using the classifier
    sign = classifier.predict([landmarks.flatten()])
    return sign[0]

def detect_hand_sign():
    # Load the pre-trained classifier
    classifier = load_classifier('classifier.pkl')

    # Define the mapping of labels to meanings
    label_to_meaning = {
        'A': 'Letter A',
        'B': 'Letter B',
        'C': 'Letter C',
        # Add more mappings for other letters here
    }

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the image
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the image
                for _, landmark in enumerate(hand_landmarks.landmark):
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                # Convert landmarks to a 1D array
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Predict hand sign
                sign = predict_hand_sign(classifier, landmarks)
                meaning = label_to_meaning.get(sign, 'Unknown')
                cv2.putText(frame, f"Sign: {meaning}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand_sign()
