import cv2
import smtplib
import numpy as np
import folium
import geocoder
import time
from email.message import EmailMessage
import os
import serial
import time

from speech_recognition import Microphone, Recognizer, UnknownValueError,RequestError


global mic
mic = True


voi = []
def callback(recognizer, audio):
    global voi
    try:
         
        voice = recognizer.recognize_google(audio, language = 'bn')
        print('You said ', voice)
        if mic == True:
            if 'বাঁচাও' in voice:
                voi.append(voice)
            else:
                voi=[]
        #print(voi)
    except UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

def Hear():
    r = Recognizer()
    m = Microphone(device_index=0)
    # start listening in the background
    with m:
        r.adjust_for_ambient_noise(m)
    print("okay")
    r.listen_in_background(m, callback)
Hear()

s = serial.Serial('COM2',9600)
if not s.isOpen():
    s.open()
print('com2 is open', s.isOpen())

def sendEmail(msge):
    mesg = EmailMessage()
    mesg['Subject'] = 'Fire detected !!!'
    mesg['From'] = 'Smart Fire Resistance'
    mesg['To'] = 'mahinshikder01@gmail.com'
    mesg.set_content(msge)
    email = 'fire'

    with open("Fire_Location.html","rb") as f:
        file_data = f.read()
        file_name = f.name
        mesg.add_attachment(file_data, maintype = "web", subtype="html",filename =file_name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login("smartfiresystem661@gmail.com", "mahinmahin99")
        server.send_message(mesg)
    

cfg = "P:\\Odommo 50\\yolov4-fire.cfg"
model = "P:\\Odommo 50\\yolov4-fire_best.weights"
net = cv2.dnn.readNetFromDarknet(cfg, model)
classes = ['Fire']
def findLoca():
    g = geocoder.ip("me")
    my_address=g.latlng
    my_map1=folium.Map(location=my_address,zoom_start=16)
    folium.CircleMarker(location=my_address,radius=50,popup="Fire").add_to(my_map1)
    folium.Marker(my_address,popup="Fire").add_to(my_map1)
    my_map1.save("P:\\Odommo 50\\Fire_Location.html")
    print("Accident Location Found.")

cap = cv2.VideoCapture("P:\\Odommo 50\\fire.mp4")

Font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
det = 0
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    
    frame_id += 1

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6:
                s.write(b'1')
                mic = True
                if len(voi)>0:
                    if 'বাঁচাও' in voi[len(voi)-1]:
                        sendEmail("Human has been dectected")
                        print("Human has been dectected. E-mail notification sent to the Rescue Team.........")
                        voi=[]
                det += 1
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    colors = np.random.uniform(0,255, size=(len(boxes),3))
    indexes = np.array(indexes)
    
    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence= str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame, label+" "+confidence,(x,y+20),Font, 1,(255,165,0),2)

    if (det in range(30,35)):
        findLoca()
        sendEmail("Fire has been dectected!!!")
        print('E-mail notification sent to the Rescue Team.........')
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), Font, 3, (255,0,0), 3)
    cv2.imshow("Live", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
