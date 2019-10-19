import cv2 as cv

# Including require Data
face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier('Data/haarcascade_smile.xml')
eye_cascade = cv.CascadeClassifier('Data/haarcascade_eye_tree_eyeglasses.xml')

# Initializing the cap variable
cap = cv.VideoCapture(0)

# Smile Detect Function
def smile_detect(img_frame, img_gray, x_coo, y_coo, wid, hei):
    smile_gray = img_gray[y_coo:y_coo + hei, x_coo:x_coo + wid]
    smile_color = img_frame[y_coo:y_coo + hei, x_coo:x_coo + wid]
    smiles = smile_cascade.detectMultiScale(smile_gray, 3, 5)
    for (sx, sy, sw, sh) in smiles:
        cv.rectangle(smile_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 5)
        cv.putText(frame, "Smile Detected", (20, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Eye Detect Function
def eye_detect(img_frame, img_gray, x_coo, y_coo, wid, hei):
    eye_gray = img_gray[y_coo:y_coo + hei, x_coo:x_coo + wid]
    eye_color = img_frame[y_coo:y_coo + hei, x_coo:x_coo + wid]
    eyes = eye_cascade.detectMultiScale(eye_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv.putText(frame, "Eyes Detected", (20, 55), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
while cap.isOpened():
    # for reading the web camera
    _, frame = cap.read()
    # for text
    cv.putText(frame, "Press 'q' to EXIT", (350, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    # converting color to gray image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # for detecting face
    faces = face_cascade.detectMultiScale(frame, 1.3, 4)
    for (x, y, w, h) in faces:
        # for drawing the rectangle around the face
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cv.putText(frame, "Face Detected", (20, 85), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        # for smile detection
        smile_detect(frame, gray, x, y, w, h)
        # for eye detection
        eye_detect(frame, gray, x, y, w, h)
    # for showing the result
    cv.imshow('Frame', frame)
    # To break the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
# releasing and destroying the created window
cap.release()
cv.destroyAllWindows()