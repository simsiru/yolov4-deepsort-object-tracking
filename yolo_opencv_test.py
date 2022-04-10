import cv2
import numpy as np
from yolo_object_detection import ObjectDetection
import time
from utils import draw_bbox

CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"]

od = ObjectDetection("yolo_models/yolov4.weights", "yolo_models/yolov4.cfg",
    nms_thr=0.4, conf_thr=0.5, img_size=416)

#cap = cv2.VideoCapture("test_videos/los_angeles.mp4")
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t1 = time.time()

    (class_ids, scores, boxes) = od.detect(frame)
    for i, box in enumerate(boxes):
        (x, y, w, h) = box
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        draw_bbox(xmin, ymin, xmax, ymax, height, frame, CLASSES, class_ids[i], scores[i],
            color=(255,255,255), rand_colors=True)
    
    fps = 1 / (time.time() - t1)


    cv2.rectangle(frame, (0, 25), (140, 0), (255, 255, 255), thickness=cv2.FILLED)
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (0, 20), cv2.FONT_HERSHEY_DUPLEX,
    0.75, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()