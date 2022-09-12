import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten


# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)


facetracker = load_model('facetracker.h5')



# Siamese L1 Distance class
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
        
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

def preprocess(file_path):
    
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image 
    img = tf.io.decode_jpeg(byte_img)
    
    # Preprocessing steps - resizing the image to be 105x105x3
    img = tf.image.resize(img, (105,105))
    # Scale image to be between 0 and 1 
    img = img / 255.0

    # Return image
    return img

# Verification Function
#os.path.join('persons')

def verify(model, detection_threshold, verification_threshold):
    # Build results array
    best_match = [0, None]
    for directory in os.listdir('persons'):
        print(directory)
        if '.' not in directory:
            results = []
            for image in os.listdir(os.path.join('persons', directory)):
                if '.jpg' not in image:
                    continue
                
                input_img = preprocess(os.path.join('temp_image', 'input_image.jpg'))
                validation_img = preprocess(os.path.join('persons', directory, image))
                
                # Make Predictions 
                result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
                results.append(result)
            
            # Detection Threshold: Metric above which a prediciton is considered positive 
            detection = np.sum(np.array(results) > detection_threshold)
            print(results)
            print(detection)
            
            # Verification Threshold: Proportion of positive predictions / total positive samples 
            verification = detection / len(os.listdir(os.path.join('persons', directory))) 
            print(verification)
            if verification > best_match[0]:
                best_match[0] = verification
                best_match[1] = directory
            #verified = verification > verification_threshold

    verified = False
    person = "No Match"
    if best_match[0] > verification_threshold:
        verified = True
        person = best_match[1]

    
    return results, verified, person

color = (0, 0, 255)
person = '_'

cap = cv2.VideoCapture(0)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
while cap.isOpened():
    _ , frame = cap.read()
    #frame = frame[int(height/2-400):int(height/2+400), int(width/2-400):int(width/2+400), :]
    frame = frame[:1000, 500:1500, :]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]

    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder 

        cv2.imwrite(os.path.join('temp_image', 'input_image.jpg'), cv2.resize(frame, (250, 250)))
        # Run verification
        results, verified, person = verify(siamese_model, 0.5, 0.5)


    if yhat[0] <= 0.5:
        person = '_'
    
    if yhat[0] > 0.5: 
        #if person == '_':
        # Save input image to application_data/input_image folder 

        cv2.imwrite(os.path.join('temp_image', 'input_image.jpg'), cv2.resize(frame, (250, 250)))
        # Run verification
        results, verified, person = verify(siamese_model, 0.5, 0.5)
        if verified:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [1000, 1000]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [1000, 1000]).astype(int)), 
                            color, 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [1000, 1000]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [1000, 1000]).astype(int),
                                    [80,0])), 
                            color, -1)
        
        # Controls the text rendered
        cv2.putText(frame, person, tuple(np.add(np.multiply(sample_coords[:2], [1000, 1000]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()