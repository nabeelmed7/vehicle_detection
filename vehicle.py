from PIL import Image
import cv2
import numpy as np
import requests  
import streamlit as st

car_cascade_src = 'cars.xml'
bus_cascade_src = 'Bus_front.xml'

def get_detection(image):
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr,cv2.COLOR_BGR2GRAY)
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    cars = car_cascade.detectMultiScale(grey, 1.1, 1)

    bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
    bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
    
    bcnt = 0
    bus = bus_cascade.detectMultiScale(grey, 1.1, 1)
    for (x,y,w,h) in bus:
        cv2.rectangle(image_arr,(x,y),(x+w,y+h),(0,255,0),2)
        bcnt += 1
        
    ccnt = 0
    if bcnt == 0:
        for (x,y,w,h) in cars:
            cv2.rectangle(image_arr,(x,y),(x+w,y+h),(255,0,0),2)
            ccnt += 1
            
    img = Image.fromarray(image_arr, 'RGB')
    result = str(ccnt) + ' cars and ' + str(bcnt) + ' buses found'
    
    return img.show(), result


st.title("Vehicle Detection App")

# ask user to upload file
uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # check if uploaded file is an image or video
    if uploaded_file.type.startswith('image'):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = Image.fromarray(image, 'RGB')
        result = get_detection(image)
        st.text(result)
    elif uploaded_file.type.startswith('video'):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = Image.fromarray(image, 'RGB')
        result = get_detection(image)
        st.text(result)
    else:
        st.error("Invalid file type. Please upload an image or video file.")





