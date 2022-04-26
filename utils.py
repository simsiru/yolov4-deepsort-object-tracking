import cv2
import numpy as np
import re
import easyocr
import os
from depth_sensor import D435i
import torch
import psycopg2 as pg
import pandas as pd

from yolo_object_detection import ObjectDetection
from facenet_pytorch import MTCNN, InceptionResnetV1
#easyocr_reader = easyocr.Reader(['en'], gpu=True) # this needs to run only once to load the model into memory


""" def recognize_plate_number_easyocr(num_plate_box, coords, area_th=0.2):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    num_plate_box = cv2.cvtColor(num_plate_box, cv2.COLOR_BGR2RGB)

    xmin, ymin, xmax, ymax = coords

    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    #num_plate_box = img[int(ymin):int(ymax), int(xmin):int(xmax)] #[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]

    plate_num = ""

    box_area = (xmax - xmin) * (ymax - ymin)

    try:
        text = easyocr_reader.readtext(num_plate_box, detail=1)

        for res in text:
            length = np.sum(np.subtract(res[0][1], res[0][0]))
            height = np.sum(np.subtract(res[0][2], res[0][1]))

            if ((length * height) / box_area) > area_th:
                plate_num += res[1]

        plate_num = re.sub('[\W_]+', '', plate_num)
    except: 
        text = None

    return plate_num """

class EasyocrNumberPlateRecognition():
    def __init__(self, area_th=0.2):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=True)
        self.letter_plate_area_ratio = area_th

    def recognize_plate_number(self, num_plate_box, coords):
        num_plate_box = cv2.cvtColor(num_plate_box, cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = coords

        plate_num = ""

        box_area = (xmax - xmin) * (ymax - ymin)

        try:
            text = self.easyocr_reader.readtext(num_plate_box, detail=1)

            for res in text:
                length = np.sum(np.subtract(res[0][1], res[0][0]))
                height = np.sum(np.subtract(res[0][2], res[0][1]))

                if ((length * height) / box_area) > self.letter_plate_area_ratio:
                    plate_num += res[1]

            plate_num = re.sub('[\W_]+', '', plate_num)
        except: 
            text = None

        return plate_num


def draw_bbox(xmin, ymin, xmax, ymax, height, frame, classes=None, score=None, class_id=None, color=(255,255,255), rand_colors=False, pass_text=False, bbox_text=None):
    if rand_colors and not pass_text:
        np.random.seed(class_id)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        color=(int(r*255),int(g*255),int(b*255))

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    if not pass_text:
        text = '{}: {:.2f}'.format(classes[class_id], score)
    else:
        text = bbox_text

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
    0.75, thickness=1)

    if (ymax + text_height) > height:
        cv2.rectangle(frame, (xmin, ymax), (xmin + text_width, ymax - text_height - baseline), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax - 4), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (xmin, ymax + text_height + baseline), (xmin + text_width, ymax), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax + text_height + 3), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)

    #return frame


def draw_bbox_tracking(xmin, ymin, xmax, ymax, height, frame, class_name, tracking_id, color=(255,255,255), rand_colors=True, info_text=""):
    if rand_colors:
        np.random.seed(tracking_id)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        color=(int(r*255),int(g*255),int(b*255))

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    text = '{} ID: {} [{}]'.format(class_name, tracking_id, info_text)

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
    0.75, thickness=1)

    if (ymax + text_height) > height:
        cv2.rectangle(frame, (xmin, ymax), (xmin + text_width, ymax - text_height - baseline), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax - 4), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (xmin, ymax + text_height + baseline), (xmin + text_width, ymax), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax + text_height + 3), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)

    #return frame


def draw_bbox_face_rec(xmin, ymin, xmax, ymax, height, frame, tracking_id, info_text=""):
    if len(info_text) != 0:
        if info_text == "HOSTILE DETECTED":
            color=(0,0,255)
        elif info_text[0] == "|":
            color=(255,255,255)
        else:
            color=(0,255,0)
    else:
        color=(255,255,255)

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    text = 'ID: {} [{}]'.format(tracking_id, info_text)

    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX,
    0.75, thickness=1)

    if (ymax + text_height) > height:
        cv2.rectangle(frame, (xmin, ymax), (xmin + text_width, ymax - text_height - baseline), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax - 4), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (xmin, ymax + text_height + baseline), (xmin + text_width, ymax), color,
        thickness=cv2.FILLED)

        cv2.putText(frame, text, (xmin, ymax + text_height + 3), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 1,
        lineType=cv2.LINE_AA)







def get_and_save_person_face_embeddings(face_embeddings_path, n_img_class = 10, save_face_depth_maps = False, save_face_img = False, face_img_path = "faces"):
    face_img_idx = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    emb_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(device=device)

    fdet_model = ObjectDetection("yolo_models/yolov4-tiny_f_.weights", "yolo_models/yolov4-tiny_1_cl.cfg", nms_thr=0.4, conf_thr=0.5, img_size=416)

    sensor = D435i(convert_to_color_map = True, width = 640, height = 480, enable_depth = True, enable_rgb = True, enable_infrared = False)


    face_embeddings, face_embeddings_labels, face_depth_maps, name_to_class_idx_map = None, None, None, None

    if os.path.exists(face_embeddings_path + "/face_embeddings.pt") or os.path.exists(face_embeddings_path + "/face_embeddings_labels.pt") \
    or os.path.exists(face_embeddings_path + "/name_to_class_idx_map.npy") or os.path.exists(face_embeddings_path + "/face_depth_maps.pt"):
        face_embeddings = torch.load(face_embeddings_path + "/face_embeddings.pt")
        face_embeddings_labels = torch.load(face_embeddings_path + "/face_embeddings_labels.pt")
        name_to_class_idx_map = list(np.load(face_embeddings_path + "/name_to_class_idx_map.npy"))

        if save_face_depth_maps:
            face_depth_maps = torch.load(face_embeddings_path + "/face_depth_maps.pt")

        face_img_idx = len(name_to_class_idx_map)
    else:
        name_to_class_idx_map = []

    curr_img_n = 0
    emb_idx = 0

    person_name = ""

    name_saving_mode = True
    emb_saving_mode = False

    dm_xmin, dm_ymin, dm_xmax, dm_ymax = 180, 80, 460, 400

    while True:
        avbl, depth, rgb, raw_depth, infrared = sensor.get_frame()
        if not avbl:
            sensor.release()
            break


        depth_with_bboxes = depth.copy()
        rgb_with_bboxes = rgb.copy()
        _, _, bboxes = fdet_model.detect(rgb)
        for box in bboxes:
            (x, y, w, h) = box
            xmin, ymin, xmax, ymax = x, y, x + w, y + h

            cv2.rectangle(depth_with_bboxes, (xmin, ymax), (xmax, ymin), (255,255,255), 2)
            cv2.rectangle(rgb_with_bboxes, (xmin, ymax), (xmax, ymin), (255,255,255), 2)

        face_depth_map = raw_depth[dm_ymin:dm_ymax, dm_xmin:dm_xmax]


        cv2.rectangle(depth_with_bboxes, (dm_xmin, dm_ymax), (dm_xmax, dm_ymin), (255,255,255), 2)

        frame_with_bboxes = np.hstack((depth_with_bboxes, rgb_with_bboxes))

        cv2.imshow('Depth and rgb camera streams', frame_with_bboxes)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            sensor.release()
            cv2.destroyAllWindows()
            break


        if cv2.waitKey(25) & 0xFF == ord("a") and name_saving_mode:
            person_name = input("Enter next person's name: ")
            print(f"Currently collecting {person_name} data")
            curr_img_n = 0

            name_to_class_idx_map.append(person_name)
            
            name_saving_mode = False
            emb_saving_mode = True
            

        if cv2.waitKey(25) & 0xFF == ord("s") and emb_saving_mode and curr_img_n <= n_img_class - 1:
            if save_face_img:
                img_path = face_img_path + "/" + person_name + "/" + str(curr_img_n) + ".jpg"
                if os.path.exists(face_img_path + "/" + person_name + "/"):
                    face = mtcnn(rgb, save_path=img_path)
                else:
                    os.mkdir(face_img_path + "/" + person_name + "/")
                    face = mtcnn(rgb, save_path=img_path)
            else:
                face = mtcnn(rgb)

            if face is None:
                continue


            embeddings = emb_model(face.unsqueeze(0).to(device)).detach().cpu()

            if emb_idx == 0 and face_img_idx == 0:
                face_embeddings = embeddings

                face_embeddings_labels = torch.tensor([face_img_idx])

                if save_face_depth_maps:
                    face_depth_maps = torch.tensor(face_depth_map.astype('int16')).unsqueeze(0)
                    #print(face_depth_maps.shape)
            else:
                face_embeddings = torch.cat((face_embeddings, embeddings), 0)

                face_embeddings_labels = torch.cat((face_embeddings_labels, torch.tensor([face_img_idx])), 0)

                if save_face_depth_maps:
                    face_depth_maps = torch.cat((face_depth_maps, torch.tensor(face_depth_map.astype('int16')).unsqueeze(0)), 0)
                    #print(face_depth_maps.shape)
            
            emb_idx += 1


            if curr_img_n == n_img_class - 1:
                name_saving_mode = True
                emb_saving_mode = False
                face_img_idx += 1


            curr_img_n += 1

            print(f"Embedding {curr_img_n} saved")

    if face_embeddings is not None and face_embeddings_labels is not None and name_to_class_idx_map is not None:
        print(face_embeddings.shape)
        print(face_embeddings_labels)
        print(name_to_class_idx_map)

        torch.save(face_embeddings, face_embeddings_path + "/face_embeddings.pt")
        torch.save(face_embeddings_labels, face_embeddings_path + "/face_embeddings_labels.pt")
        np.save(face_embeddings_path + "/name_to_class_idx_map.npy", name_to_class_idx_map)

        if save_face_depth_maps and face_depth_maps is not None:
            print(face_depth_maps.shape)
            torch.save(face_depth_maps, face_embeddings_path + "/face_depth_maps.pt")



def get_and_save_person_face_depth_maps(face_embeddings_path, n_img_class = 10):
    sensor = D435i(convert_to_color_map = True, width = 640, height = 480, enable_depth = True, enable_rgb = True, enable_infrared = False)

    face_depth_maps, name_to_class_idx_map = None, None

    dm_exists = False

    if os.path.exists(face_embeddings_path + "/name_to_class_idx_map.npy"):
        name_to_class_idx_map = list(np.load(face_embeddings_path + "/name_to_class_idx_map.npy"))

        if os.path.exists(face_embeddings_path + "/face_depth_maps.pt"):
            face_depth_maps = torch.load(face_embeddings_path + "/face_depth_maps.pt")
            dm_exists = True
        #face_dm_len = len(name_to_class_idx_map)
    else:
        print('No embeddings data found')
        return

    dm_xmin, dm_ymin, dm_xmax, dm_ymax = 180, 80, 460, 400

    curr_dm = 0

    for i, name in enumerate(name_to_class_idx_map):
        print(f"Currently saving for {name}")
        while True:
            avbl, depth, rgb, raw_depth, infrared = sensor.get_frame()
            if not avbl:
                sensor.release()
                break

            depth_with_bboxes = depth.copy()
            rgb_with_bboxes = rgb.copy()

            face_depth_map = raw_depth[dm_ymin:dm_ymax, dm_xmin:dm_xmax]

            cv2.rectangle(depth_with_bboxes, (dm_xmin, dm_ymax), (dm_xmax, dm_ymin), (255,255,255), 2)

            frame_with_bboxes = np.hstack((depth_with_bboxes, rgb_with_bboxes))

            cv2.imshow('Depth and rgb camera streams', frame_with_bboxes)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                sensor.release()
                cv2.destroyAllWindows()
                break

            if cv2.waitKey(25) & 0xFF == ord("s"):
                if not dm_exists:
                    face_depth_maps = torch.tensor(face_depth_map.astype('int16')).unsqueeze(0)
                    curr_dm += 1
                    dm_exists = True
                else:
                    face_depth_maps = torch.cat((face_depth_maps, torch.tensor(face_depth_map.astype('int16')).unsqueeze(0)), 0)
                    curr_dm += 1

                print(f"{curr_dm} saved")

                if curr_dm % n_img_class == 0:
                    curr_dm = 0
                    break

    if face_depth_maps is not None:
        print(face_depth_maps.shape)
        torch.save(face_depth_maps, face_embeddings_path + "/face_depth_maps.pt")





def delete_person_face_embeddings(face_embeddings_path, person_name):
    face_embeddings = torch.load(face_embeddings_path + "/face_embeddings.pt")
    face_embeddings_labels = torch.load(face_embeddings_path + "/face_embeddings_labels.pt")
    name_to_class_idx_map = list(np.load(face_embeddings_path + "/name_to_class_idx_map.npy"))

    face_depth_maps = torch.load(face_embeddings_path + "/face_depth_maps.pt")

    if person_name not in name_to_class_idx_map:
        print("No person with that name")
        return

    for i, name in enumerate(name_to_class_idx_map):
        if name == person_name:
            person_idx = i
            break
    name_to_class_idx_map.remove(person_name)

    delete_idx = []
    for i in range(face_embeddings_labels.shape[0]):
        if face_embeddings_labels[i] == person_idx:
            delete_idx.append(i)

    n_img_class = len(delete_idx)

    #face_embeddings_labels = face_embeddings_labels[face_embeddings_labels != person_idx]
    face_embeddings_labels = torch.cat((face_embeddings_labels[:delete_idx[0]], face_embeddings_labels[delete_idx[0] + n_img_class:] - 1), 0)

    face_embeddings = torch.cat((face_embeddings[:delete_idx[0]], face_embeddings[delete_idx[0] + n_img_class:]), 0)

    face_depth_maps = torch.cat((face_depth_maps[:delete_idx[0]], face_depth_maps[delete_idx[0] + n_img_class:]), 0)

    print(face_embeddings_labels)
    print(face_embeddings.shape)
    print(face_depth_maps.shape)
    print(name_to_class_idx_map)

    torch.save(face_embeddings, face_embeddings_path + "/face_embeddings.pt")
    torch.save(face_embeddings_labels, face_embeddings_path + "/face_embeddings_labels.pt")
    np.save(face_embeddings_path + "/name_to_class_idx_map.npy", name_to_class_idx_map)

    torch.save(face_depth_maps, face_embeddings_path + "/face_depth_maps.pt")



def view_face_embeddings(face_embeddings_path):
    if os.path.exists(face_embeddings_path + "/face_embeddings.pt") and os.path.exists(face_embeddings_path + "/face_embeddings_labels.pt") \
    and os.path.exists(face_embeddings_path + "/name_to_class_idx_map.npy") and os.path.exists(face_embeddings_path + "/face_depth_maps.pt"):
        face_embeddings = torch.load(face_embeddings_path + "/face_embeddings.pt")
        face_embeddings_labels = torch.load(face_embeddings_path + "/face_embeddings_labels.pt")
        name_to_class_idx_map = list(np.load(face_embeddings_path + "/name_to_class_idx_map.npy"))

        face_depth_maps = torch.load(face_embeddings_path + "/face_depth_maps.pt")

        print(face_embeddings_labels)
        print(face_embeddings.shape)
        print(face_depth_maps.shape)
        print(name_to_class_idx_map)
    else:
        print("Files not found")
        return


class DBInterface():
    def __init__(self, password, username='postgres', hostname='localhost', database='face_recognition', port_id=5432):
        self.host = hostname
        self.dbname = database
        self.username = username
        self.password = password
        self.port = port_id
        

    def execute_sql_script(self, sql_script, return_result = False):
        conn = None
        cur = None
        df = None

        try:
            with pg.connect(
                host = self.host,
                dbname = self.dbname,
                user = self.username,
                password = self.password,
                port = self.port) as conn:

                if return_result:
                    df = pd.read_sql_query(sql_script, conn)
                else:
                    with conn.cursor(cursor_factory=pg.extras.DictCursor) as cur:
                        cur.execute(sql_script)

                        #if len(cur.fetchall()) > 0:
                        #    df = cur.fetchall()
                
                    #conn.commit()
        except Exception as error:
            print(error)
        finally:
            #if cur is not None:
            #    cur.close()
            if conn is not None:
                conn.close()

        if return_result:
            return df
        