import tensorflow as tf
import cv2
import numpy as np
import time
from utils import draw_bbox, draw_bbox_tracking

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]) #2048 3072
    except RuntimeError as e:
        print(e)


""" CLASSES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
    "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
    "scissors","teddy bear","hair drier","toothbrush"] """

CLASSES =  {1: "person",2: "bicycle",3: "car",4: "motorcycle",5: "airplane",6: "bus",7: "train",
    8: "truck",9: "boat",10: "traffic light",11: "fire hydrant",13: "stop sign",14: "parking meter",
    15: "bench",16: "bird",17: "cat",18: "dog",19: "horse",20: "sheep",21: "cow",22: "elephant",23: "bear",
    24: "zebra",25: "giraffe",27: "backpack",28: "umbrella",31: "handbag",32: "tie",33: "suitcase",34: "frisbee",
    35: "skis",36: "snowboard",37: "sports ball",38: "kite",39: "baseball bat",40: "baseball glove",
    41: "skateboard",42: "surfboard",43: "tennis racket",44: "bottle",46: "wine glass",47: "cup",48: "fork",
    49: "knife",50: "spoon",51: "bowl",52: "banana",53: "apple",54: "sandwich",55: "orange",56: "broccoli",
    57: "carrot",58: "hot dog",59: "pizza",60: "donut",61: "cake",62: "chair",63: "couch",64: "potted plant",
    65: "bed",67: "dining table",70: "toilet",72: "tv",73: "laptop",74: "mouse",75: "remote",76: "keyboard",
    77: "cell phone",78: "microwave",79: "oven",80: "toaster",81: "sink",82: "refrigerator",84: "book",85: "clock",
    86: "vase",87: "scissors",88: "teddy bear",89: "hair drier",90: "toothbrush"}


#TRACK_CLASSES = ["car","motorcycle","bus","truck"]
TRACK_CLASSES = ["person"]


#CLASSES = {1:"License_plate"}

#dict_keys(['raw_detection_scores', 'detection_anchor_indices', 'detection_multiclass_scores', 'raw_detection_boxes', 'detection_boxes', 'detection_classes', 'num_detections', 'detection_scores'])



if __name__ == "__main__":

    #model = tf.saved_model.load('ssdmobilenetv2_model/saved_model_coco')
    model = tf.saved_model.load('efficientdetd1_model/saved_model_coco')


    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    #initialize deep sort object
    model_filename = 'deep_sort/mars-small128.pb'
    encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)



    #cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("test_videos/los_angeles.mp4")
    #cap = cv2.VideoCapture("test_videos/long_traffic_video.mp4")
    #cap = cv2.VideoCapture("test_videos/test_video.mp4")
    cap = cv2.VideoCapture("test_videos/test.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    min_score = 0.5

    roi_select = True
    roi_x = 0
    roi_y = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if roi_select:
            roi = cv2.selectROI(frame)
            roi_x, roi_y, width, height = roi[0], roi[1], roi[2], roi[3]
            roi_select = False
            cv2.destroyAllWindows()

        frame = frame[roi_y:roi_y+height, roi_x:roi_x+width]



        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break

        t1 = time.time()


        detections = model(np.expand_dims(frame, axis=0))


        frame_with_bboxes = frame.copy()

        boxes, scores, names = [], [], []
        for i, id in enumerate(detections['detection_classes'][0]):
            if detections['detection_scores'][0][i] < min_score:
                break
            #if int(id) == 1:
            if CLASSES[int(id)] in TRACK_CLASSES:
                koord = detections['detection_boxes'][0][i]
                boxes.append([int(koord[1]*width), int(koord[0]*height), int(koord[3]*width)-int(koord[1]*width), int(koord[2]*height)-int(koord[0]*height)])
                scores.append(detections['detection_scores'][0][i])
                names.append(CLASSES[int(id)])


            #koord = detections['detection_boxes'][0][i]
            #xmin, ymin, xmax, ymax = int(koord[1]*width), int(koord[0]*height), int(koord[3]*width), int(koord[2]*height)
            #draw_bbox(xmin, ymin, xmax, ymax, height, frame_with_bboxes, CLASSES, int(id), detections['detection_scores'][0][i],
            #color=(255,255,255), rand_colors=True)
                


        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track

            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height

            #bbox=[xmin, ymin, xmax, ymax]
            #print("bbox: {}, class name: {}, tracking ID: {}".format(bbox, class_name, tracking_id))
            draw_bbox_tracking(xmin, ymin, xmax, ymax, height, frame_with_bboxes, class_name, tracking_id,
            color=(255,255,255), rand_colors=True)




        period = time.time() - t1
        frequency = 1 / period

        #cv2.rectangle(frame_with_bboxes, (0, height), (140, height-25), (255, 255, 255), thickness=cv2.FILLED)
        cv2.rectangle(frame_with_bboxes, (0, 25), (140, 0), (255, 255, 255), thickness=cv2.FILLED)
        #cv2.putText(frame_with_bboxes, 'FPS: {:.2f}'.format(frequency), (0, height-5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        #    lineType=cv2.LINE_AA)
        cv2.putText(frame_with_bboxes, 'FPS: {:.2f}'.format(frequency), (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
            lineType=cv2.LINE_AA)


        cv2.imshow('Object tracking with deepSORT', frame_with_bboxes)


