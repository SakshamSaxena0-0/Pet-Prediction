##### Importing Libraries #####
import tensorflow as tf
import numpy as np
from keras.preprocessing import image

##### Load the trained CNN model from file #####
cnn = tf.keras.models.load_model('cat_dog_model.h5')

##### Load and preprocess the test image #####
###### Resize image to 64x64 pixels as per model requirements #####
test_image = image.load_img('image_path', target_size = (64,64))

##### Convert image to numerical array #####
test_image = image.img_to_array(test_image)

#####Adding batch dimension to image#####
test_image = np.expand_dims(test_image, axis = 0)

##### Make prediction using the model #####
result = cnn.predict(test_image)

##### Model outputs 1 for dog, 0 for cat ##### 
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

##### Displaying the prediction and confidence score #####
print(f"Prediction: {prediction}")
print("Confidence: {abs(result[0][0] * 100):.2r}%")
