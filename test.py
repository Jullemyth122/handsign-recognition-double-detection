import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3
import speech_recognition as sr


def takeCommand():
 
    r = sr.Recognizer()
 
    # from the speech_Recognition module
    # we will use the Microphone module
    # for listening the command
    with sr.Microphone() as source:
        print('Listening')
         
        # seconds of non-speaking audio before
        # a phrase is considered complete
        r.pause_threshold = 0.7
        audio = r.listen(source)
         
        # Now we will be using the try and catch
        # method so that if sound is recognized
        # it is good else we will have exception
        # handling
        try:
            print("Recognizing")
             
            # for Listening the command in indian
            # english we can also use 'hi-In'
            # for hindi recognizing
            Query = r.recognize_google(audio, language='en-in')
            print("the command is printed=", Query)
             
        except Exception as e:
            print(e)
            print("Say that again sir")
            return "None"
         
        return Query

def speak(audio):
     
    engine = pyttsx3.init()
    # getter method(gets the current value
    # of engine property)
    voices = engine.getProperty('voices')
     
    # setter method .[0]=male voice and
    # [1]=female voice in set Property.
    engine.setProperty('voice', voices[0].id)
     
    # Method for the speaking of the assistant
    engine.say(audio) 
     
    # Blocks while processing all the currently
    # queued commands
    engine.runAndWait()

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=5)
classifier = Classifier("C:\Python CV\HandSignDetection\Model\keras_model.h5","C:\Python CV\HandSignDetection\Model\labels.txt")

offset = 20
imgSize = 300

folder = "C:\Python CV\HandSignDetection\Data\C"
counter = 0

labels = ["A","B","C","D","F"]

Change = True

# while Change:

while True:

    success, img = cap.read()

    imgOutput = img.copy()
    hands,img = detector.findHands(img)

    imgWhite = np.ones((imgSize,imgSize,3),np.uint8) * 255

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y+h+offset,x - offset :x+w + offset]

        imgCropShape = imgCrop.shape
        # imgWhite[0:imgCrop.shape[0],0:imgCropShape[1]] = imgCrop

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)

            imgWhite[:,wGap:wCal + wGap] = imgResize
            
            prediction,index = classifier.getPrediction(imgWhite,draw=True)
            # print(prediction,index)

        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)

            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction,index = classifier.getPrediction(imgWhite,draw=True)


        
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset),(0, 66, 235),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),(cv2.FONT_HERSHEY_COMPLEX),2,(52, 171, 235),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(0, 66, 235),2)

        # if (labels[index] == "A"):
        #     speak("This is A")
        # if (labels[index] == "B"):
        #     speak("This is B")
        # if (labels[index] == "C"):
        #     speak("This is C")
        # if (labels[index] == "D"):
        #     speak("This is D")
        # if (labels[index] == "F"):
        #     speak("This is F")
        # cv2.imshow("ImageCrop",imgCrop)
        # cv2.imshow("ImageWhite",imgWhite)
    

    cv2.imshow("Image",img)
    cv2.imshow("Image",imgOutput)
    key = cv2.waitKey(1)