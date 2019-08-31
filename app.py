from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import csv
import datetime
from camera_pi import Camera
import face_recog as face
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainer/trainer.yml')
cascadePath = "C:/Users/Kamil/Documents/face/friday webapp/webapp/Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX


#generet

names = ['None', 'Akshay', 'Chaitanya', 'Ilza', 'Z', 'W'] 

app = Flask(__name__)

@app.route('/', methods = ["GET","POST"])
def contact():
      if "checkout_button" in request.form: #name="checkout_button" is the value request.form['checkout_button] = checkout  (value)
        print(request.form["checkout_button"])
        return render_template('recog.html')
      elif "checkin_button" in request.form:
        print('checkin button')
        return render_template('recog.html')
      else:
          return render_template('index.html')

@app.route('/recog')
def recog():
    """Video Strea  ming Home Page."""
    return render_template('recog.html')

#def gen(camera):
def gen():
    recognize = True
    id = 0
    initial = ""
    tempid = ""
    i = 0
    """Video Streaming Generator Function"""
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while recognize:
        ret, img = cam.read()
        img = cv2.flip(img, -1) # Flip vertically
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            tempid = id
            print(str(tempid) + "  " + str(i))
            
            if tempid == initial and tempid != "":
                i += 1
                initial = tempid
                if i ==30:
                    return render_template('/')
            else:
                i = 0
                initial = tempid    
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', img)[1].tobytes() + b'\r\n')
'''
cam.release()
    Akshay  27
Akshay  28
Akshay  29
Debugging middleware caught exception in streamed response at a point where response headers were already sent.
Traceback (most recent call last):
  File "/usr/lib/python3/dist-packages/werkzeug/wsgi.py", line 870, in __next__
    return self._next()
  File "/usr/lib/python3/dist-packages/werkzeug/wrappers.py", line 82, in _iter_encoded
    for item in iterable:
  File "/home/pi/webapp/app.py", line 72, in gen
    return render_template('/')
  File "/usr/lib/python3/dist-packages/flask/templating.py", line 133, in render_template
    ctx.app.update_template_context(context)
AttributeError: 'NoneType' object has no attribute 'app'
'''
    
       
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute fo an img tag."""
    return Response(gen(),#gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/welcome')
def welcome():
    with open('csv_files/mycsv.csv', 'a') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Johnson', datetime.date.today().strftime('%d/%m/%Y'), datetime.datetime.now().time().strftime('%H:%M:%S')])
    return 'Welcome Johnson'    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=7000, threaded=True)
    

