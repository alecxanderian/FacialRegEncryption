"""
Author: Alec C.  Simardeep K.
Title: Face-Based Encryption Software
Date: April 17, 2025

Description:
This program allows a user to register their face and use it as a biometric encryption key.
It uses dlib's face recognition model to extract a face embedding and then derives a SHA-256
hash from it to serve as a cryptographic key. The system can encrypt and decrypt files 
based on face matching. It also displays real-time video feed and interaction prompts. The comment
headers below show a basic gist on what each function doe.
"""

import cv2
import dlib
import os
import pickle
import numpy as np
import hashlib
import base64
from cryptography.fernet import Fernet, InvalidToken
from imutils import face_utils

# Load face detection and embedding models
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()

# File paths and constants
EMBED_PATH = "face_embedding.pkl"
KEY_PATH = "face_key.key"
IMG_PATH = "reference_face.png"
TEST_FILE = "secret.txt"
ENCRYPTED_FILE = "secret.enc"
UNLOCK_DISTANCE_THRESHOLD = .2  # match threshold

# Video setup
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Extract face embedding and bounding box from an image
def get_face_embedding(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector(gray, 1)
    if not faces:
        return None, None

    face = faces[0]
    shape = predictor(rgb, face)
    descriptor = face_encoder.compute_face_descriptor(rgb, shape)
    embedding = np.array(descriptor)

    (x, y, w, h) = face_utils.rect_to_bb(face)
    return embedding, (y, x + w, y + h, x)

# Convert an embedding into a SHA-256 cryptographic key
def embedding_to_key(embedding):
    flat = embedding.tobytes()
    return hashlib.sha256(flat).digest()

# Encrypt a file using the derived key
def encrypt_file(key, filename=TEST_FILE, out_file=ENCRYPTED_FILE):
    with open(filename, "rb") as f:
        data = f.read()
    fernet = Fernet(base64.urlsafe_b64encode(key))
    encrypted = fernet.encrypt(data)
    with open(out_file, "wb") as f:
        f.write(encrypted)
    print("File encrypted and saved as", out_file)

# Decrypt a file using the derived key
def decrypt_file(key, enc_file=ENCRYPTED_FILE):
    try:
        with open(enc_file, "rb") as f:
            data = f.read()
        fernet = Fernet(key)
        decrypted = fernet.decrypt(data)
        print("Decrypted content:\n", decrypted.decode())
    except InvalidToken:
        print("Decryption failed: Invalid key (face matched distance, but keys don't align).")
    except Exception as e:
        print("Error during decryption:", e)

# Check if two face embeddings are close enough (below distance threshold)
def is_face_match(live_emb, saved_emb, threshold=UNLOCK_DISTANCE_THRESHOLD):
    distance = np.linalg.norm(live_emb - saved_emb)
    print(f"Face distance: {distance:.4f}")
    return distance < threshold, distance

print("Press 'C' to register face")
print("Press 'E' to encrypt file")
print("Press 'D' to decrypt file")
print("Press 'Q' to quit")

# Main loop for capturing video and handling key presses
while True:
    ret, frame = video_capture.read()
    display = frame.copy()

    embedding, face_location = get_face_embedding(frame)

    if face_location:
        top, right, bottom, left = face_location
        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Encryption System", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c') and embedding is not None:
        with open(EMBED_PATH, "wb") as f:
            pickle.dump(embedding, f)
        cv2.imwrite(IMG_PATH, frame)

        exact_key = base64.urlsafe_b64encode(embedding_to_key(embedding))
        with open(KEY_PATH, "wb") as f:
            f.write(exact_key)

        print("Face registered and encryption key saved.")

    elif key == ord('e') and embedding is not None:
        if os.path.exists(KEY_PATH):
            with open(KEY_PATH, "rb") as f:
                stored_key = f.read()
            encrypt_file(base64.urlsafe_b64decode(stored_key))
        else:
            print("No encryption key found. Press 'C' to register face.")

    elif key == ord('d') and embedding is not None:
        if os.path.exists(EMBED_PATH) and os.path.exists(KEY_PATH):
            with open(EMBED_PATH, "rb") as f:
                saved_embedding = pickle.load(f)
            matched, distance = is_face_match(embedding, saved_embedding)

            if matched:
                with open(KEY_PATH, "rb") as f:
                    stored_key = f.read()
                decrypt_file(stored_key)
            else:
                print(f"Face mismatch — distance too high ({distance:.4f}). Decryption blocked.")
        else:
            print("Missing face data or key. Press 'C' to register.")

video_capture.release()
cv2.destroyAllWindows()
