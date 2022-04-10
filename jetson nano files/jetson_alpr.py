import jetson.inference
import jetson.utils

import cv2
import time
import numpy as np
from collections import Counter

#from utils import recognize_plate_number_easyocr
from utils import EasyocrNumberPlateRecognition


#net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
""" net = jetson.inference.detectNet("ssd-mobilenet-v2",
[
"--model=ssd-mobilenet.onnx",
"--class_labels=labels.txt",
"--input-blob=input_0",
"--output-cvg=scores",
"--output-bbox=boxes"],
threshold=0.5)

camera = jetson.utils.videoSource("/dev/video0") # '/dev/video0' for V4L2 """

#display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file
#display = jetson.utils.videoOutput("my_video.mp4")



#while display.IsStreaming():
#cap = cv2.VideoCapture(0)
""" while True:
     cuda_frame = camera.Capture()
     frame = jetson.utils.cudaToNumpy(cuda_frame)

     #ret, frame = cap.read()

          
     detections = net.Detect(cuda_frame)
          
     for detection in detections:
          xmin, ymin, xmax, ymax = detection.Left, detection.Top, detection.Right, detection.Bottom
          #print(f"Top: {ymin}, Bottom: {ymax}, Left: {xmin}, Right: {xmax}")
          #print(detection.Confidence)



     #display.Render(img)
     #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS())) """









if __name__ == "__main__":
     net = jetson.inference.detectNet("ssd-mobilenet-v2",
     [
     "--model=ssd-mobilenet.onnx",
     "--class_labels=labels.txt",
     "--input-blob=input_0",
     "--output-cvg=scores",
     "--output-bbox=boxes"],
     threshold=0.5)

     camera = jetson.utils.videoSource("/dev/video0")

     num_plate_data_dir = "saved_number_plates/"

     ocr = EasyocrNumberPlateRecognition(area_th=0.2)

     while True:
          t1 = time.time()


          plate_num_buf = []
          while len(plate_num_buf) < 10:
               cuda_frame = camera.Capture()
               frame = jetson.utils.cudaToNumpy(cuda_frame).copy()

               detections = net.Detect(cuda_frame)

               plate_number = ""
               for detection in detections:
                    xmin, ymin, xmax, ymax = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
                    num_plate_img = frame[ymin:ymax, xmin:xmax]

                    #plate_number = recognize_plate_number_easyocr(num_plate_img, (xmin, ymin, xmax, ymax), area_th=0.2)
                    plate_number = ocr.recognize_plate_number(num_plate_img, (xmin, ymin, xmax, ymax))
                    break

               plate_num_buf.append(plate_number)


          plate_num_buf = list(filter(None, plate_num_buf))
          data = Counter(plate_num_buf)
          if len(plate_num_buf) > 0:
               plate_number = data.most_common(1)[0][0]
               plate_number = plate_number.upper()
               print(plate_number)
               #cv2.imwrite(num_plate_data_dir + plate_number + ".jpg", num_plate_img)
               

          period = time.time() - t1
          frequency = 1 / period
          print(frequency)


















"""Object Detection DNN - locates objects in an image
 
Examples (jetson-inference/python/examples)
     detectnet-console.py
     detectnet-camera.py
 
__init__(...)
     Loads an object detection model.
 
     Parameters:
       network (string) -- name of a built-in network to use
                           see below for available options.
 
       argv (strings) -- command line arguments passed to imageNet,
                         see below for available options.
 
       threshold (float) -- minimum detection threshold.
                            default value is 0.5
 
detectNet arguments: 
  --network NETWORK     pre-trained model to load, one of the following:
                            * pednet (default)
                            * multiped
                            * facenet
                            * ssd-mobilenet-v1
                            * ssd-mobilenet-v2
                            * ssd-inception-v2
                            * coco-airplane
                            * coco-bottle
                            * coco-chair
                            * coco-dog
  --model MODEL         path to custom model to load (.caffemodel, .uff, or .onnx)
  --prototxt PROTOTXT   path to custom prototxt to load (for .caffemodel only)
  --class_labels LABELS path to text file containing the labels for each class
  --threshold THRESHOLD minimum threshold for detection (default is 0.5)
  --input_blob INPUT    name of the input layer (default is 'data')
  --output_cvg COVERAGE name of the coverge output layer (default is 'coverage')
  --output_bbox BOXES   name of the bounding output layer (default is 'bboxes')
  --mean_pixel PIXEL    mean pixel value to subtract from input (default is 0.0)
  --batch_size BATCH    maximum batch size (default is 1)"""