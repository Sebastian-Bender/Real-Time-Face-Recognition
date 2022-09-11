import cv2
import os
from PIL import Image
import uuid
import time

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# Make the directories
if not os.path.isdir('data'):
    os.makedirs('data')
if not os.path.isdir(POS_PATH):
    os.makedirs(POS_PATH)
if not os.path.isdir(NEG_PATH):
    os.makedirs(NEG_PATH)
if not os.path.isdir(ANC_PATH):
    os.makedirs(ANC_PATH)


# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)
while cap.isOpened(): 
    ret, frame = cap.read()
   
    #frame = frame[400:400+250,800:800+250, :]
    frame = frame[int(height/2-400):int(height/2+400), int(width/2-400):int(width/2+400), :]
    
    # Collect anchors 
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path 
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        print(imgname)
        # reshape image to 250x250 and save
        cv2.imwrite(imgname, cv2.resize(frame, (250, 250)))
    
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path 
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        print(imgname)
        # reshape image to 250x250 and save
        cv2.imwrite(imgname, cv2.resize(frame, (250, 250)))

    # Collect Image for verification
    if cv2.waitKey(1) & 0XFF == ord('v'):
        # Create the unique file path 
        imgname = os.path.join('persons', '{}.jpg'.format(uuid.uuid1()))
        print(imgname)
        # reshape image to 250x250 and save
        cv2.imwrite(imgname, cv2.resize(frame, (250, 250)))
    
    # Show image back to screen
    cv2.imshow('Image Collection', frame)
    
    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
        
# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()