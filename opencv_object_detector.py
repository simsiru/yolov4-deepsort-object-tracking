import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path, cfg_path, nms_thr=0.4, conf_thr=0.5, img_size=416, enable_cuda=True):
        print("Loading opencv dnn model")

        self.nmsThreshold = nms_thr
        self.confThreshold = conf_thr
        #self.image_size = 608
        self.image_size = img_size

        net = cv2.dnn.readNet(weights_path, cfg_path)
        #net = cv2.dnn.readNetFromDarknet(weights_path, cfg_path)

        if enable_cuda:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(net)

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)

