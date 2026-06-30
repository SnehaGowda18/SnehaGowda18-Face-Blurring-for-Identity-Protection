import cv2

# Load Haar Cascade from OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Create Full Screen Window
window_name = "Face Blurring for Identity Protection"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("========================================")
print(" Face Blurring Started")
print(" Press 'Q' to Exit")
print("========================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Flip image like a mirror
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    # Blur every detected face
    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size != 0:

            blurred = cv2.GaussianBlur(face, (99, 99), 30)

            frame[y:y+h, x:x+w] = blurred

            # Optional rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:   # Q or ESC
        break

cap.release()
cv2.destroyAllWindows()