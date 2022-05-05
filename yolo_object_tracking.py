import tensorflow as tf
import cv2
import numpy as np
import time
from collections import Counter
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections

from embeddings_classifier import EmbeddingsClassifier
from depth_map_classifier import DepthMapsClassifier
from opencv_object_detector import ObjectDetection
from utils import draw_bbox, draw_bbox_tracking, draw_bbox_face_rec, EasyocrNumberPlateRecognition

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]) #2048 3072
    except RuntimeError as e:
        print(e)


def yolo_object_tracking_with_apps(roi_select = False, use_sensor = False, do_lpr = False,
    save_lp = False, do_face_rec = False, do_face_rec_with_depth_map = False, video_file_path=None):

    classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella",
        "handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
        "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
        "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
        "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
        "scissors","teddy bear","hair drier","toothbrush"]
    track_classes = ["car","bus","truck","motorcycle","person"]

    #classes = ["vehicle registration plate","car","bus","truck","motorcycle"]

    #classes = ["vehicle registration plate"]
    #track_classes = ["vehicle registration plate"]

    """ classes = ["Face"]
    track_classes = ["Face"] """


    #model = ObjectDetection("yolo_models/yolov4.weights", "yolo_models/yolov4.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)
    #model_lp = ObjectDetection("yolo_models/yolov4_lp.weights", "yolo_models/yolov4_lp.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)
    #model_lp_v = ObjectDetection("yolo_models/yolov4_lp_v.weights", "yolo_models/yolov4_lp_v.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)

    model = ObjectDetection("yolo_models/yolov4-tiny.weights", "yolo_models/yolov4-tiny.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)


    #DeepSORT setup
    max_cosine_distance = 0.7
    nn_budget = None
    model_filename = 'deep_sort/mars-small128.pb'
    encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    #ROI setup
    roi_x = 0
    roi_y = 0


    #D435i setup
    if use_sensor:
        from depth_sensor import D435i
        sensor = D435i(convert_to_color_map = True, width = 640, height = 480, enable_depth = True, enable_rgb = True, enable_infrared = False)

        width = sensor.width
        height = sensor.height

    #Camera stream setup
    else:
        if video_file_path is not None:
            cap = cv2.VideoCapture(video_file_path)
        else:
            cap = cv2.VideoCapture(0)

        #cap = cv2.VideoCapture("test_videos/traffic3.mp4")
        #cap = cv2.VideoCapture("test_videos/people1.mp4")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    #LPR setup
    prev_id = -1
    n_det = 10
    max_n_obj = 50
    last_obj_to_del = 0

    if do_lpr:
        ocr = EasyocrNumberPlateRecognition(area_th=0.2)
        num_pl_cl = 0
        lp_obj_dict = {}
        num_plate_data_dir = "saved_number_plates/"

        model_lp = ObjectDetection("yolo_models/yolov4-tiny_lp.weights", "yolo_models/yolov4-tiny_1_cl.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)


    #FR setup
    if do_face_rec_with_depth_map:
        face_depth_map_classifier = DepthMapsClassifier(len(face_idx_to_class_map), )
        face_depth_map_classifier.load_state_dict(torch.load('depth_map_classifier_model/dm_classifier.pth', map_location="cuda:0"))
        face_depth_map_classifier.eval().to(device)

        dm_xmin, dm_ymin, dm_xmax, dm_ymax = 180, 80, 460, 400

    if do_face_rec and use_sensor:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        face_obj_dict = {}

        face_idx_to_class_map = np.load("face_embeddings_data/name_to_class_idx_map.npy")#, allow_pickle=True)

        face_embeddings_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        face_embeddings_classifier = EmbeddingsClassifier(len(face_idx_to_class_map))
        face_embeddings_classifier.load_state_dict(torch.load('embeddings_classifier_model/emb_classifier.pth', map_location="cuda:0"))
        face_embeddings_classifier.eval().to(device)

        mtcnn = MTCNN(device=device)

        face_min_rec_prob = 70

        model = ObjectDetection("yolo_models/yolov4-tiny_f_.weights", "yolo_models/yolov4-tiny_1_cl.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)
        classes = ["Face"]
        track_classes = ["Face"]


    rec_proc_state_map = {0: "|",
                          1: "||",
                          2: "|||",
                          3: "||||",
                          4: "|||||",
                          5: "||||||",
                          6: "|||||||",
                          7: "||||||||",}


    while True:
        if use_sensor:
            avbl, depth, frame, raw_depth, _ = sensor.get_frame()
            if not avbl:
                sensor.release()
                break

            if do_face_rec_with_depth_map:
                #depth_with_bboxes = depth.copy()
                raw_depth = raw_depth[dm_ymin:dm_ymax, dm_xmin:dm_xmax]
                cv2.rectangle(depth, (dm_xmin, dm_ymax), (dm_xmax, dm_ymin), (255,255,255), 2)
        else:
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
            if use_sensor:
                sensor.release()
            else:
                cap.release()
            cv2.destroyAllWindows()
            break

        t1 = time.time()


        frame_with_bboxes = frame.copy()


        boxes, scores, names = [], [], []

        (cl_ids, probs, bboxes) = model.detect(frame)
        for i, box in enumerate(bboxes):
            if classes[cl_ids[i]] in track_classes:
                boxes.append([box[0], box[1], box[2], box[3]])
                scores.append(probs[i])
                names.append(classes[cl_ids[i]])
                


        # Obtain all the detections for the given frame.
        boxes = np.array(boxes) 
        names = np.array(names)
        scores = np.array(scores)
        features = np.array(encoder(frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        n_objects = len(tracker.tracks)
        curr_obj = []

        # Obtain info from the tracks
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 

            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track

            curr_obj.append(tracking_id)

            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height


            #License plate recognition
            if do_lpr:
                if len(lp_obj_dict) > max_n_obj:
                    if last_obj_to_del not in lp_obj_dict:
                        last_obj_to_del += 1
                    else:
                        del lp_obj_dict[last_obj_to_del]
                        #lp_dict.pop(tracking_id)
                if len(lp_obj_dict) == 0:
                    last_obj_to_del = tracking_id

                if tracking_id > prev_id:
                    prev_id = tracking_id
                    lp_obj_dict[tracking_id] = [1, [], []]


                if lp_obj_dict[tracking_id][0] <= n_det:
                    bbox_frame = frame[ymin:ymax, xmin:xmax]
                    plate_number = ""

                    #plate_number = ocr.recognize_plate_number(bbox_frame, (xmin, ymin, xmax, ymax))

                    (cl_ids, probs, bboxes) = model_lp.detect(bbox_frame)
                    for i, box in enumerate(bboxes):
                        if cl_ids[i] == num_pl_cl:
                            bbox_x, bbox_y, bbox_w, bbox_h = box[0], box[1], box[2], box[3]
                            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h
                            num_plate_img = bbox_frame[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
                            plate_number = ocr.recognize_plate_number(num_plate_img, (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
                            break

                    lp_obj_dict[tracking_id][1].append(plate_number)

                    idx = int(lp_obj_dict[tracking_id][0] * (len(rec_proc_state_map) / n_det))
                    lp_obj_dict[tracking_id][2] = rec_proc_state_map[idx - 1 if idx == len(rec_proc_state_map) else idx]

                    lp_obj_dict[tracking_id][0] += 1

                if lp_obj_dict[tracking_id][0] == n_det + 1:
                    temp_buf = list(filter(None, lp_obj_dict[tracking_id][1]))
                    temp_buf = Counter(temp_buf)
                    if len(temp_buf) > 0:
                        plate_number = temp_buf.most_common(1)[0][0]
                        plate_number = plate_number.upper()
                        lp_obj_dict[tracking_id][2] = plate_number
                        if save_lp:
                            cv2.imwrite(num_plate_data_dir + plate_number + ".jpg", num_plate_img)
                    else:
                        lp_obj_dict[tracking_id][2] = "NO LP"

                    lp_obj_dict[tracking_id][0] += 1

            #----------------------------


            #Face recognition
            if do_face_rec:
                if len(face_obj_dict) > max_n_obj:
                    if last_obj_to_del not in face_obj_dict:
                        last_obj_to_del += 1
                    else:
                        del face_obj_dict[last_obj_to_del]
                        #lp_dict.pop(tracking_id)
                if len(face_obj_dict) == 0:
                    last_obj_to_del = tracking_id

                if tracking_id > prev_id:
                    prev_id = tracking_id
                    face_obj_dict[tracking_id] = [1, [], [], []]


                if face_obj_dict[tracking_id][0] <= n_det:
                    #bbox_frame = frame[ymin:ymax, xmin:xmax]
                    bbox_frame = mtcnn(frame)

                    if bbox_frame is not None:

                        #face_embeddings = face_embeddings_model(bbox_frame)
                        #face_embeddings = face_embeddings_classifier(face_embeddings)

                        face_embeddings = face_embeddings_model(bbox_frame.unsqueeze(0).to(device))
                        face_embeddings = face_embeddings_classifier(face_embeddings).detach().cpu()
                        face_embeddings = torch.nn.functional.softmax(face_embeddings)

                        face_obj_dict[tracking_id][1].append(face_embeddings.numpy().copy()[0])

                        idx = int(face_obj_dict[tracking_id][0] * (len(rec_proc_state_map) / n_det))
                        face_obj_dict[tracking_id][2] = rec_proc_state_map[idx - 1 if idx == len(rec_proc_state_map) else idx]

                        face_obj_dict[tracking_id][0] += 1


                        if do_face_rec_with_depth_map:
                            out = face_depth_map_classifier(torch.tensor(raw_depth).unsqueeze(0)).detach().cpu()
                            out = torch.nn.functional.softmax(out)
                            face_obj_dict[tracking_id][3].append(out.numpy().copy()[0])


                if face_obj_dict[tracking_id][0] == n_det + 1:
                    #res = face_embeddings_classifier(face_obj_dict[tracking_id][1])
                    #mean = np.mean(res, axis=0)
                    #print(face_obj_dict[tracking_id][1])
                    
                    mean = np.mean(face_obj_dict[tracking_id][1], axis=0)
                    #print(mean)
                    idx = np.argmax(mean)
                    #print(idx)
                    prob = int(np.max(mean)*100)
                    #print(prob)

                    if prob >= face_min_rec_prob:
                        face_obj_dict[tracking_id][2] = face_idx_to_class_map[idx]
                    else:
                        face_obj_dict[tracking_id][2] = "HOSTILE DETECTED"


                    if do_face_rec_with_depth_map:
                        mean = np.mean(face_obj_dict[tracking_id][3], axis=0)
                        #print(mean)
                        idx = np.argmax(mean)
                        #print(idx)
                        prob = int(np.max(mean)*100)
                        #print(prob)
                        if prob >= face_min_rec_prob:
                            #face_obj_dict[tracking_id][2] = face_idx_to_class_map[idx]
                            print(f'Depth map ID: {face_idx_to_class_map[idx]}')
                        else:
                            #face_obj_dict[tracking_id][2] = "HOSTILE DETECTED"
                            print(f'Depth map unknown')




                    face_obj_dict[tracking_id][0] += 1

            #----------------------------


            if do_lpr and tracking_id in lp_obj_dict:
                draw_bbox_tracking(xmin, ymin, xmax, ymax, height, frame_with_bboxes, class_name,
                tracking_id, color=(255,255,255), rand_colors=True, info_text=lp_obj_dict[tracking_id][2])

            if do_face_rec and tracking_id in face_obj_dict:
                draw_bbox_face_rec(xmin, ymin, xmax, ymax, height, frame_with_bboxes, tracking_id,
                info_text=face_obj_dict[tracking_id][2])

            if not do_lpr and not do_face_rec:
                draw_bbox_tracking(xmin, ymin, xmax, ymax, height, frame_with_bboxes, class_name,
                tracking_id, color=(255,255,255), rand_colors=True)


        fps = 1 / (time.time() - t1)

        #cv2.rectangle(frame_with_bboxes, (0, 25), (140, 0), (255, 255, 255), thickness=cv2.FILLED)
        cv2.rectangle(frame_with_bboxes, (0, 46), (280, 0), (255, 255, 255), thickness=cv2.FILLED)

        cv2.putText(frame_with_bboxes, 'FPS: {:.2f}'.format(fps), (0, 20), cv2.FONT_HERSHEY_DUPLEX,
        0.75, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        cv2.putText(frame_with_bboxes, 'Number of objects: {}'.format(n_objects), (0, 40), cv2.FONT_HERSHEY_DUPLEX,
        0.75, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        if do_face_rec_with_depth_map:
            cv2.imshow('Object tracking with deepSORT', np.hstack((depth, frame_with_bboxes)))
        else:
            cv2.imshow('Object tracking with deepSORT', frame_with_bboxes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start YOLOv4 object tracking')
    parser.add_argument('--roi', type=bool, default=False, help='Select region of interest in video stream')
    parser.add_argument('-s', '--sensor', type=bool, default=False, help='Whether or not to use a realsense D435i sensor')
    parser.add_argument('--lpr', type=bool, default=False, help='Activate license plate recognition application')
    parser.add_argument('--lpr_save_img', type=bool, default=False, help='Whether or not to save license plate image when license plate recognition is active')
    parser.add_argument('--fr', type=bool, default=False, help='Activate face recognition application')
    parser.add_argument('--fr_depth_map', type=bool, default=False, help='Whether or not to use face depth maps when face recognition and sensor is active')
    parser.add_argument('-v', '--video_file_path', default=None, help='Use a video for tracking and if the path is not provided use a webcam')
    args = parser.parse_args()

    yolo_object_tracking_with_apps(args.roi, args.sensor, args.lpr,
    args.lpr_save_img, args.fr, args.fr_depth_map, args.video_file_path)

