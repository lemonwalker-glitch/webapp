import cv2
import numpy as np
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
#for each person enter the ID
face_id = input('\n enter user id end press <return> ==>   ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")
#Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        #Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+h])
        cv2.imshow('image', img)
        
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 50: #Take 30 face samples and stop video
        break
    
#Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()