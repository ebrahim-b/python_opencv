import cv2
import numpy as np


plate_classifier=cv2.CascadeClassifier("haarcascade_plate_number.xml")


cam = cv2.VideoCapture("plate.mp4")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Car Plate", frame)

    k = cv2.waitKey(1)
    if k & 0xFF == ord('q'):
        break
    elif k & 0xFF == ord('s'):
        img_name = "opencv_frame_{}.png".format(img_counter)
        plate_img = frame

        imgGray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        plate_rects= plate_classifier.detectMultiScale(imgGray, 1.1 , 5)

        for (x,y,w,h) in plate_rects:
            crop_img = plate_img[y:y+h, x:x+w]
            cv2.rectangle(plate_img,
                            (x,y),
                            (x+w,y+h),
                            (255,255,255),
                            3)
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, crop_img)
            print("{} written!".format(img_name))
            img_counter += 1



cam.release()

cv2.destroyAllWindows()