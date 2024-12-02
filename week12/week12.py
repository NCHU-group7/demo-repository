import cv2
import numpy as np
import os
from deepface import DeepFace

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Release the camera
cap.release()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Create the week12 directory if it doesn't exist
output_dir = 'week12'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the original image
cv2.imwrite(os.path.join(output_dir, 'original.jpg'), frame)

# Rotate 60 degrees to the left
(h, w) = frame.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 60, 1.0)
rotated = cv2.warpAffine(frame, M, (w, h))
cv2.imwrite(os.path.join(output_dir, 'rotated_60_left.jpg'), rotated)

# Flip horizontally and vertically
flipped_hv = cv2.flip(frame, -1)
cv2.imwrite(os.path.join(output_dir, 'flipped_hv.jpg'), flipped_hv)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, 'grayscale.jpg'), gray)

# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imwrite(os.path.join(output_dir, 'hsv.jpg'), hsv)

# Apply mosaic effect
def mosaic(img, scale=0.1):
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

mosaic_img = mosaic(frame)
cv2.imwrite(os.path.join(output_dir, 'mosaic.jpg'), mosaic_img)

print("Images have been saved successfully.")

# Reinitialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(output_dir, 'output.avi'), fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame for facial expressions
    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']

        # Draw the dominant emotion on the frame
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error analyzing frame: {e}")

    # Write the frame to the video file
    out.write(frame)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and video writer
cap.release()
out.release()
cv2.destroyAllWindows()