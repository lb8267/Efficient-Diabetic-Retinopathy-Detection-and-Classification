import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow


grad_model = tf.keras.models.Model([mdl.input], [mdl.get_layer('conv2d_1').output, mdl.output])


for i in range(0, N_test_images):
  label_ground_truth = label_matrix_appended_test[i]
  image_ground_truth = image_matrix_appended_test[i]
  image_ground_truth = image_ground_truth.reshape(1, 256, 256, 3) 
  prediction = mdl.predict(image_ground_truth)
  target_class = np.argmax(prediction)


  
  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(image_ground_truth)
      loss = predictions[:, target_class]
  
  output = conv_outputs[0]
  grads = tape.gradient(loss,conv_outputs)[0]
  gate_f = tf.cast(output > 0, 'float32')
  gate_r = tf.cast(grads > 0, 'float32')
  guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
  weights = tf.reduce_mean(guided_grads, axis=(0, 1))
  weights_gradcam = tf.reduce_mean(grads, axis=(0, 1))
  cam = np.ones(output.shape[0:2], dtype=np.float32)
  cam_gradcam = np.ones(output.shape[0:2], dtype=np.float32)

  for index, w in enumerate(weights):
    cam += w * output[:,:,index]

  for index, w in enumerate(weights_gradcam):
    cam_gradcam += w * output[:,:,index]

  #Heatmap Visualization
  cam = cv2.resize(cam.numpy(), (256, 256))
  cam_gradcam = cv2.resize(cam_gradcam.numpy(), (256, 256))

  cam = np.maximum(cam, 0)
  cam_gradcam = np.maximum(cam_gradcam, 0)

  heatmap = (cam - cam.min()) / (cam.max() - cam.min())
  heatmap_gradcam = (cam_gradcam - cam_gradcam.min()) / (cam_gradcam.max() - cam_gradcam.min())

  image_ground_truth = image_ground_truth.reshape(256,256,3)

  cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
  output_image = cv2.addWeighted(cv2.cvtColor(image_ground_truth.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
  
  cam_gradcam = cv2.applyColorMap(np.uint8(255*heatmap_gradcam), cv2.COLORMAP_JET)
  output_image_gradcam = cv2.addWeighted(cv2.cvtColor(image_ground_truth.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam_gradcam, 1.0, 0)
  
 
  path = '/content/drive/My Drive/GradImages/'+ 'Grad-CAM_' +  str(i) + '.png'

  if (label_ground_truth == 0):
    label_ground_truth = 'Normal'
  else if(label_ground_truth == 1):
    label_ground_truth = 'Mild'
  else if(label_ground_truth == 2):
    label_ground_truth = 'Moderate'
  else if(label_ground_truth == 3):
    label_ground_truth = 'Severe'
  else :
    label_ground_truth = 'Advanced'

  if (target_class == 0):
    target_class = 'Normal'
  else if(target_class == 1):
    target_class = 'Mild'
  else if(target_class == 2):
    target_class = 'Moderate'
  else if(target_class == 3):
    target_class = 'Severe' 
  else:
    target_class = 'Advanced'
  
  b,g,r = cv2.split(output_image)       
  output_image = cv2.merge([r,g,b])
    
  b,g,r = cv2.split(output_image_gradcam)       
  output_image_gradcam = cv2.merge([r,g,b])  

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 3, 1)
  plt.xlabel('Label Ground Truth:'+ ' ' + str(label_ground_truth))
  plt.imshow(image_ground_truth)
  plt.title('Image Ground Truth')
  
  plt.subplot(1, 3, 2)
  plt.imshow(output_image_gradcam)
  plt.xlabel('Label Predicted:'+  ' ' + str(target_class))
  plt.title('GradCAM')
  plt.savefig(path, bbox_inches='tight')
