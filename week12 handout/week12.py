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

# Record video from the webcam
cap = cv2.VideoCapture(0) 
fourcc = cv2.VideoWriter_fourcc(*'XVID') # compression codec
out = cv2.VideoWriter('week12 handout/output.avi', fourcc, 20.0, (640, 480) )

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load the face detection model

start_time = cv2.getTickCount()
duration = 10  # Duration in seconds

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Apply mosaic to faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (w // 10, h // 10))
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = face

    out.write(frame)

    # Check if the duration has been reached
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time > duration:
        break

cap.release() # Release the capture
out.release() # Release the video writer
cv2.destroyAllWindows() # Close all OpenCV windows