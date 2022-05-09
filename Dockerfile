FROM opencv4.5.5-cuda:11.3.0-cudnn8-devel-ubuntu20.04

RUN mv /usr/local/lib/python3.8/site-packages/cv2/ /usr/local/lib/python3.8/dist-packages/

COPY . /home/yolov4_tracking_apps/

RUN cd /home/yolov4_tracking_apps/ && pip3 install -r requirements.txt --no-deps

WORKDIR /home/yolov4_tracking_apps

#ENTRYPOINT python3 mot_server.py
#ENTRYPOINT ["python3", "mot_server.py"]