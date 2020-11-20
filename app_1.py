import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_blue = np.array([38,86,0])
    upper_blue = np.array([121,255,255])
    mask = cv2.inRange(hsv,lower_blue,upper_blue)

    cent , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for c in cent:
        cv2.drawContours(frame, c, -1, (255, 0, 0), 2)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()