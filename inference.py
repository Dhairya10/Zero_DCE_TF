import numpy as np
import tensorflow as tf
import keras.backend as K
from PIL import Image

input_image_path = ''

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_dce.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loading Input Image
original_img = Image.open(input_image_path)
original_size = (np.array(original_img).shape[1], np.array(original_img).shape[0])
original_img = original_img.resize((512,512), Image.ANTIALIAS) 
original_img = (np.asarray(original_img)/255.0)

img_lowlight = Image.open(input_image_path)        
img_lowlight = img_lowlight.resize((512,512), Image.ANTIALIAS)
img_lowlight = (np.asarray(img_lowlight, dtype=np.float32)/255.0) 
img_lowlight = np.expand_dims(img_lowlight, 0)

# Test the model on input image
input_shape = input_details[0]['shape']
# input_data = np.array(img_lowlight, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], img_lowlight)
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

r1, r2, r3, r4, r5, r6, r7, r8 = output_data[:,:,:,:3], output_data[:,:,:,3:6], output_data[:,:,:,6:9], output_data[:,:,:,9:12], output_data[:,:,:,12:15], output_data[:,:,:,15:18], output_data[:,:,:,18:21], output_data[:,:,:,21:24]
x = original_img + r1 * (K.pow(original_img,2)-original_img)
x = x + r2 * (K.pow(x,2)-x)
x = x + r3 * (K.pow(x,2)-x)

enhanced_image_1 = x + r4*(K.pow(x,2)-x)
x = enhanced_image_1 + r5*(K.pow(enhanced_image_1,2)-enhanced_image_1)		
x = x + r6*(K.pow(x,2)-x)	
x = x + r7*(K.pow(x,2)-x)

enhance_image = x + r8*(K.pow(x,2)-x)
enhance_image = tf.cast((enhance_image[0,:,:,:] * 255), dtype=np.uint8)
enhance_image = Image.fromarray(enhance_image.numpy())
enhance_image = enhance_image.resize(original_size, Image.ANTIALIAS)
enhance_image.save('output_rs.jpg')

