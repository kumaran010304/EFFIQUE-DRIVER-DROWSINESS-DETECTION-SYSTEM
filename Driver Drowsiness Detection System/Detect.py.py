import cv2
import os
import numpy as np
import time
import RPi.GPIO as GPIO

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
BUZZER_PIN = 23  # GPIO pin for the buzzer
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Load Haar cascade classifiers and Keras model
face_cascade = cv2.CascadeClassifier('haarcascadefiles/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haarcascadefiles/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haarcascadefiles/haarcascade_righteye_2splits.xml')
model = load_model('models/cnncat2.h5')

# Initialize variables
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2

# Main loop
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Detect eyes
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    # Classify eyes using Keras model
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = np.argmax(model.predict(r_eye), axis=-1)

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = np.argmax(model.predict(l_eye), axis=-1)

    # Update drowsiness score
    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure score is within bounds
    score = max(0, score)

    # Check for drowsiness
    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(2)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        except:
            pass

    # Update thickness of rectangle
    if thicc < 16:
        thicc += 2
    else:
        thicc -= 2
    thicc = max(2, thicc)

    # Draw rectangle around frame
    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display frame
    cv2.imshow('frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
