import cv2 
import numpy as np

def display_green_objects(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([65, 100, 35])
    upper_green = np.array([98, 255, 190])

    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    kernel = np.ones((15,15), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    combined_frame = np.vstack((frame, np.repeat(green_mask[:, :, np.newaxis], 3, axis=2)))
    return combined_frame

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("nan camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

while True:
    # frame = bgr
    ret, frame = cap.read()

    if not ret:
        break

    #workspace
    frame = display_green_objects(frame)

    # print(frame.shape)
    scale = 0.5
    frame = cv2.resize(frame, (-1, -1), fx=scale, fy=scale)

    cv2.imshow('Camera 1', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()