import cv2

# Capture image from the webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read() # ret for success/failure, frame for the image

# Release the capture
cap.release()

if ret:
    # Rotate 90 degrees to the right
    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('week12 handout/rotated.jpg', rotated)

    # Horizontal flip
    flipped = cv2.flip(frame, 1)
    cv2.imwrite('week12 handout/flipped.jpg', flipped)

    # Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('week12 handout/gray.jpg', gray)
else:
    print("Failed to capture image")
# IRL
# Real-time face detection with mosaic effect
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Reinitialize the video capture
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('week12 handout/output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Apply mosaic effect to detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (w//10, h//10))
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = face

    # Write the frame to the output video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Face Detection with Mosaic', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()