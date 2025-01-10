import cv2
from random import randrange
data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#img=cv2.imread('rdj.jpg')
webcam=cv2.VideoCapture(0)
while True:
    succesful_frame_read, frame = webcam.read()

    #camera
    greyscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #face detect
    face_coordinates=data.detectMultiScale(greyscaled_img)
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)), 2)

    cv2.imshow('clever programmer face detector',frame)
    key=cv2.waitKey(1)
    if key==81 or key==123:
        break
webcam.release()