import cv2
import os
import numpy as np

# Initialize the face recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Path to store face data and the model
dataset_path = 'face_dataset'
model_path = 'face_model.yml'

# Check if the dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load existing model if available, otherwise skip loading
label_dict = {}
if os.path.exists(model_path):
    recognizer.read(model_path)
    print("Loaded existing face recognition model.")
    # Load the label dictionary from the dataset
    for file in os.listdir(dataset_path):
        if file.endswith('.jpg'):
            name = file.split('_')[0]
            if name not in label_dict:
                label_dict[name] = len(label_dict)
else:
    print("No existing model found. Starting fresh.")

# Function to save face data for a new person
def save_new_face(name, face_region):
    count = len([f for f in os.listdir(dataset_path) if f.startswith(name)])
    face_img_path = f'{dataset_path}/{name}_{count + 1}.jpg'
    cv2.imwrite(face_img_path, face_region)

    # Append to label dictionary and retrain model
    faces = []
    labels = []
    label_id = 0

    for file in os.listdir(dataset_path):
        if file.endswith('.jpg'):
            name = file.split('_')[0]
            if name not in label_dict:
                label_dict[name] = label_id
                label_id += 1

            img_path = os.path.join(dataset_path, file)
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(face_img)
            labels.append(label_dict[name])

    if len(faces) > 0 and len(labels) > 0:
        recognizer.train(faces, np.array(labels))
        recognizer.save(model_path)
        print(f"Added and trained on new face: {name}")
    else:
        print("No data available to train the model.")

# Main function to detect and recognize faces
def recognize_and_add_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]

            # Try to recognize the face
            try:
                label, confidence = recognizer.predict(face_region)
                if confidence < 100:  # Confidence threshold
                    name = [key for key, value in label_dict.items() if value == label][0]
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    raise ValueError  # If not recognized, fall to except
            except:
                # Face not recognized, ask for a name and store the face
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Recognition', frame)
                cv2.waitKey(1)

                name = input("Enter name for this person: ")
                if name not in label_dict:
                    label_dict[name] = len(label_dict)
                save_new_face(name, face_region)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == '__main__':
    recognize_and_add_faces()
