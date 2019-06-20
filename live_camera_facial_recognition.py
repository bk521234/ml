#Load trained model
from keras.models import load_model
from keras_vggface import utils
import cv2
import numpy as np
from keras_vggface.utils import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
train_path = 'data/'
image_size = 224
batch_size = 5

train_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(image_size,image_size),
                        batch_size=batch_size,
                        class_mode='sparse',
                        color_mode='rgb')




image_size = 224
device_id = 0 #camera_device id 

#model = load_model('my faces.h5')
model = load_model('vgg_face.h5')

print(model)

#make labels according to your dataset folder 
#labels = dict(jonny_depp=0,rowan_atkinson=1, robertdownyJr=2, jackma=3, json_statham=4) #and so on
#print(labels)

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(device_id)

#labels = dict((v,k) for k,v in labels.items())
labels = (train_generator.class_indices)
print(labels)
labels = dict((v,k) for k,v in labels.items())
print(labels)

while camera.isOpened():
    ok, cam_frame = camera.read()
    if not ok:
        break
    
    gray_img=cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
    faces= cascade_classifier.detectMultiScale(gray_img, minNeighbors=5)

    for (x,y,w,h) in faces:
        cv2.rectangle(cam_frame,(x,y),(x+w,y+h),(255,255,0),2)
        roi_color = cam_frame [y:y+h, x:x+w]
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        roi_color = cv2.resize(roi_color, (image_size, image_size))
        image = roi_color.astype(np.float32, copy=False)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image, version=1) # or version=2
        preds = model.predict(image)
        predicted_class=np.argmax(preds,axis=1)

        
        print(labels)
        print(predicted_class)
        name = [labels[k] for k in predicted_class]

        cv2.putText(cam_frame,str(name), 
                    (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        
    cv2.imshow('video image', cam_frame)
    key = cv2.waitKey(30)
    if key == 27: # press 'ESC' to quit
        break

camera.release()
cv2.destroyAllWindows()
