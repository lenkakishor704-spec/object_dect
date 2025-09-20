# import cv2

# # Load Haar Cascade file (make sure face_ref.xml exists in same folder)
# face_ref = cv2.CascadeClassifier("face_ref.xml")
# camera = cv2.VideoCapture(0)

# def face_detection(frame):
#     optimized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # fixed
#     face = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=5)
#     return face

# def drawer_box(frame):
#     for x, y, w, h in face_detection(frame):
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# def close_window():
#     camera.release()
#     cv2.destroyAllWindows()
#     exit()


import cv2
import face_recognition
import os
import numpy as np

# === Step 1: Load known faces from face_data folder ===
KNOWN_FACES_DIR = "face_data"
known_encodings = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"✅ Loaded {filename} as {name}")
        else:
            print(f"⚠ No face found in {filename}")

# === Step 2: Haar Cascade for quick detection ===
face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

def face_detection(frame):
    """Detect faces using Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_ref.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return face

def drawer_box(frame):
    """Draw bounding box + recognize face"""
    faces = face_detection(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for (x, y, w, h) in faces:
        # Convert Haar box → face_recognition format
        top, right, bottom, left = y, x + w, y + h, x
        face_location = [(top, right, bottom, left)]

        # Get face encodings
        encodings = face_recognition.face_encodings(rgb_frame, face_location)

        name = "Unknown"
        color = (0, 0, 255)  # Red = unknown

        if encodings:
            face_encoding = encodings[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_names[best_match_index]
                color = (0, 255, 0)  # Green = known

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("❌ Camera not accessible.")
            break

        drawer_box(frame)
        cv2.imshow("Kishor AI - Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
            break

if __name__ == "__main__":
    main()


