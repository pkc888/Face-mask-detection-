# from tkinter import N
# from turtle import width
from cv2 import VideoCapture
from flask import Flask, redirect, render_template, Response, url_for
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from imutils.video import VideoStream
import imutils
import time

app=Flask(__name__) 



def detect_and_predict_mask(frame,faceNet,maskNet):
    #grab the dimensions of the frame and then construct a blob

    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))
    
    faceNet.setInput(blob)
    detections=faceNet.forward()
    print(detections.shape)
    
    #initialize our list of faces, their corresponding locations and list of predictions
    
    faces=[]
    locs=[]
    preds=[]
    
    
    for i in range(0,detections.shape[2]):
        confidence=detections[0,0,i,2]
    
    
        if confidence>0.5:
        #we need the X,Y coordinates
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype('int')
        
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))
        
            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
        
            faces.append(face)
            locs.append((startX,startY,endX,endY))
        
        #only make a predictions if atleast one face was detected
        if len(faces)>0:
            faces=np.array(faces,dtype='float32')
            preds=maskNet.predict(faces,batch_size=32)
        
        return (locs,preds)

        

prototxtPath = r"D:\covid 19 facemask detection\Mask_Detection\deploy.prototxt"
weightsPath = r"D:\covid 19 facemask detection\Mask_Detection\res10_300x300_ssd_iter_140000.caffemodel"

faceNet=cv2.dnn.readNet(prototxtPath,weightsPath)

maskNet=load_model(r"D:\covid 19 facemask detection\Mask_Detection\mask_detector.model")

# Url = 'http://192.168.0.105:4747/video'
# camera=cv2.VideoCapture(Url)

camera=cv2.VideoCapture(0)

count=0
def generate_frames(camera,count=0):
    while True:
        success,frame=camera.read()
        # if not success:
        #     break
        # else:
        #     ret, buffer=cv2.imencode('.jpg',frame)
        #     frame=buffer.tobytes()
        # yield(b'--frame\r\n' b'content-Type/jpeg\r\n\r\n' + frame + b'\r\n')

        #grab the frame from the threaded video stream and resize it
        #to have a maximum width of 800 pixels
        # frame=camera.read()

        frame=imutils.resize(frame, width=None)
        
        #detect faces in the frame and preict if they are waring masks or not
        (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
        
        #loop over the detected face locations and their corrosponding loactions
      
        for (box,pred) in zip(locs,preds):
            (startX,startY,endX,endY)=box
            (mask,withoutMask)=pred
            
            #determine the class label and color we will use to draw the bounding box and text
            label='Mask' if mask>withoutMask else 'No Mask'
            color=(0,255,0) if label=='Mask' else (0,0,255)
            
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            #display the label and bounding boxes
            cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_COMPLEX,0.45,color,2)
            
            cv2.rectangle(frame,(startX,startY),(endX,endY),color,2)
            
            if (color==(0,0,255)):
                cv2.imwrite("D:\images\Frame"+str(count)+".jpg",frame)
                count=count+1
            
            
        #show the output frame
        # cv2.imshow("Frame",frame)
        # key=cv2.waitKey(1) & 0xFF
        
        ret, jpeg = cv2.imencode('.jpeg', frame)
        # ret, jpeg = cv2.imshow('.jpg', frame)
        
        
        
        frame = jpeg.tobytes()

        yield(b'--frame\r\n' b'content-Type/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
            


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/video')
def video():
    global camera
    return Response(generate_frames(camera,count=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(port=5501,debug=True)
