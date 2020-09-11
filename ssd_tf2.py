# -*- coding: utf-8 -*-
"""ssd.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iGeiyqkHnpEmlwMBrlE2Lr3kw87-jtxA
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # info
import sys
import tensorflow as tf
import scipy.misc
import numpy as np
import six
import time
import multiprocessing
from six import BytesIO
#from google.colab.patches import cv2_imshow
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import gc
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# tf.config.gpu.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
#
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3024)])
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path (this can be local or on colossus)

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

detect_fn  = tf.saved_model.load("tf_pretrainedModel/ssd_mobilenet_v2_coco_2018_03_29/saved_model")
print("model size: ", str(sys.getsizeof(detect_fn)))
print(list(detect_fn.signatures.keys()))
infer = detect_fn.signatures["serving_default"]
print(infer.structured_outputs)
print("infer size: ", sys.getsizeof(infer))
gc.collect()
im = cv2.imread("test_images/image1.jpg")
im = cv2.resize(im,(300,300))
##cv2_imshow(im)
input_tensor = np.expand_dims(im, axis=0)
start_time = time.time()
detections = infer(tf.constant(input_tensor))
end_time = time.time()
t1 = end_time - start_time
print("inference time: ", str(end_time - start_time))

# print(detections)
bb = detections['detection_boxes']
c = detections['detection_classes'][0].numpy().astype(np.int32)
s = detections['detection_scores'][0].numpy()
print(c)
print(s)
# def inference(i,return_dict):
#   print ('number: ',str(i))
#   detect_fn  = tf.saved_model.load("tf_pretrainedModel/ssd_mobilenet_v2_coco_2018_03_29/saved_model")
#   print("model size: ", str(sys.getsizeof(detect_fn)))
#   print(list(detect_fn.signatures.keys()))
#   infer = detect_fn.signatures["serving_default"]
#   print(infer.structured_outputs)
#   print("infer size: ", sys.getsizeof(infer))
#   gc.collect()
#   # return_dict[0]=infer
#   im = cv2.imread("test_images/image1.jpg")
#   ##cv2_imshow(im)
#   # image_np = load_image_into_numpy_array("/content/drive/My Drive/Colab Notebooks/dog.jpg")
#   input_tensor = np.expand_dims(im, axis=0)
#   start_time = time.time()
#   detections = infer(tf.constant(input_tensor))
#   end_time = time.time()
#   t1 = end_time - start_time
#   print("inference time: ", str(end_time - start_time))
#
#   # print(detections)
#   bb = detections['detection_boxes']
#   c = detections['detection_classes'][0].numpy().astype(np.int32)
#   s = detections['detection_scores'][0].numpy()
#   print(c)
#   print(s)
#
# manager = multiprocessing.Manager()
# return_dict = manager.dict()
# process_eval = multiprocessing.Process(target=inference, args=(0,return_dict))
# process_eval.start()
# process_eval.join()
