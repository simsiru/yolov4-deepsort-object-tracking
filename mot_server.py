import cv2
import numpy as np
import time
from collections import Counter
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import socket
import pickle
import struct

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
#from deep_sort import generate_detections
from deep_sort.pytorch_reid_feature_extractor import Extractor, get_features

from embeddings_classifier import EmbeddingsClassifier
from depth_map_classifier import DepthMapsClassifier
from opencv_object_detector import ObjectDetection
from utils import draw_bbox, draw_bbox_tracking, draw_bbox_face_rec, EasyocrNumberPlateRecognition


def yolov4_mot(roi_select = False, do_lpr = False, save_lp = False,
do_face_rec = False, do_face_rec_with_depth_map = False, port = 9999):

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

    """ classes = ["Face"]
    track_classes = ["Face"] """

    model = ObjectDetection("yolo_models/yolov4-tiny.weights",
    "yolo_models/yolov4-tiny.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)


    #DeepSORT setup
    max_cosine_distance = 0.2
    nn_budget = 100
    max_iou_distance = 0.7
    max_age = 70
    n_init = 3
    extractor = Extractor('deep_sort/ckpt.t7', use_cuda=True)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_iou_distance, max_age, n_init)


    #ROI setup
    roi_x = 0
    roi_y = 0


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

        model_lp = ObjectDetection("yolo_models/yolov4-tiny_lp.weights",
        "yolo_models/yolov4-tiny_1_cl.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)


    #FR setup
    if do_face_rec:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        face_obj_dict = {}

        face_idx_to_class_map = np.load("face_embeddings_data/name_to_class_idx_map.npy")#, allow_pickle=True)

        face_embeddings_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        face_embeddings_classifier = EmbeddingsClassifier(len(face_idx_to_class_map))
        face_embeddings_classifier.load_state_dict(torch.load('embeddings_classifier_model/emb_classifier.pth',
        map_location="cuda:0"))
        face_embeddings_classifier.eval().to(device)

        mtcnn = MTCNN(device=device)

        face_min_rec_prob = 80

        model = ObjectDetection("yolo_models/yolov4-tiny_f_.weights",
        "yolo_models/yolov4-tiny_1_cl.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)
        classes = ["Face"]
        track_classes = ["Face"]

    if do_face_rec_with_depth_map:
        face_depth_map_classifier = DepthMapsClassifier(len(face_idx_to_class_map), (320, 280))
        face_depth_map_classifier.load_state_dict(torch.load('depth_map_classifier_model/dm_classifier.pth',
        map_location="cuda:0"))
        face_depth_map_classifier.eval().to(device)

        dm_xmin, dm_ymin, dm_xmax, dm_ymax = 180, 80, 460, 400


    rec_proc_state_map = {0: "|",
                          1: "||",
                          2: "|||",
                          3: "||||",
                          4: "|||||",
                          5: "||||||",
                          6: "|||||||",
                          7: "||||||||",}


    
    #Server setup
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_name  = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    socket_address = (host_ip, port)
    server_socket.bind(socket_address)
    server_socket.listen(5)
    print("LISTENING AT:",socket_address)
    data = b""
    payload_size = struct.calcsize("Q")


    while True:
        client_socket, addr = server_socket.accept()
        print('CLIENT CONNECTED FROM:', addr)

        while client_socket:
            try:
                while len(data) < payload_size:
                    packet = client_socket.recv(4*1024) # 4K
                    if not packet:
                        break
                    data+=packet
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q",packed_msg_size)[0]   
                while len(data) < msg_size:
                    data += client_socket.recv(4*1024)
                frame_data = data[:msg_size]
                data  = data[msg_size:]
                frame = pickle.loads(frame_data)

                if do_face_rec_with_depth_map:
                    while len(data) < payload_size:
                        packet = client_socket.recv(4*1024) # 4K
                        if not packet:
                            break
                        data+=packet
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("Q",packed_msg_size)[0]   
                    while len(data) < msg_size:
                        data += client_socket.recv(4*1024)
                    frame_data = data[:msg_size]
                    data  = data[msg_size:]
                    raw_depth = pickle.loads(frame_data)

                    raw_depth = raw_depth[dm_ymin:dm_ymax, dm_xmin:dm_xmax]
            except Exception:
                break

            height, width = frame.shape[0], frame.shape[1]


            if roi_select:
                roi = cv2.selectROI(frame)
                roi_x, roi_y, width, height = roi[0], roi[1], roi[2], roi[3]
                roi_select = False
                cv2.destroyAllWindows()

            frame = frame[roi_y:roi_y+height, roi_x:roi_x+width]



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

            #features = np.array(encoder(frame, boxes))
            #print(features.shape, get_features(extractor, boxes, frame).shape)
            features = get_features(extractor, boxes, frame)

            detections = [Detection(bbox, score, class_name, feature) 
            for bbox, score, class_name, feature in zip(boxes, scores, names, features)]

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
                                plate_number = ocr.recognize_plate_number(num_plate_img,
                                (bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax))
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
                                out = face_depth_map_classifier(torch.tensor((raw_depth / 65_535.0).astype('float32')).unsqueeze(0).unsqueeze(0).to(device))
                                out = torch.nn.functional.softmax(out.detach().cpu())
                                face_obj_dict[tracking_id][3].append(out.numpy().copy()[0])


                    if face_obj_dict[tracking_id][0] == n_det + 1:
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
                                print(f'Depth map ID: Unknown')




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

            #if do_face_rec_with_depth_map:
            #    a = pickle.dumps(np.hstack((depth, frame_with_bboxes)))
            #else:
            a = pickle.dumps(frame_with_bboxes)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)


        print(f'CLIENT FROM: {addr} DISCONNECTED')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start YOLOv4 object tracking')
    parser.add_argument('--roi', type=bool, default=False,
    help='Select region of interest in video stream')
    parser.add_argument('--lpr', type=bool, default=False,
    help='Activate license plate recognition application')
    parser.add_argument('--lpr_save_img', type=bool, default=False,
    help='Whether or not to save license plate image when license plate recognition is active')
    parser.add_argument('--fr', type=bool, default=False,
    help='Activate face recognition application')
    parser.add_argument('--fr_depth_map', type=bool, default=False,
    help='Whether or not to use face depth maps when face recognition and sensor is active')
    parser.add_argument('-p', '--port', type=int, default=9999,
    help='Choose a port for a server to listen at, if not provided port 9999 will be used')
    args = parser.parse_args()

    yolov4_mot(args.roi, args.lpr, args.lpr_save_img, args.fr, args.fr_depth_map, args.port)
