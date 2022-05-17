# YOLOv4 object tracking with DeepSORT

YOLOv4 and DeepSORT object tracking application in automatic license plate recognition and face recognition. Face recognition system also uses face depth maps, extracted using D435i sensor, as a complementary verification to rgb images.

## Usage with Docker

### Windows

First run the docker container for mot server. Enter a port to use default is 9999 and whether to activate license plate recognition (--lpr) or face recognition (--fr).
The region of interests can be activated (--roi) to select a part of a screen.

```
docker run --rm -it -p PORT:PORT --gpus all ladonq/mot:0.0.1 python3 mot_server.py -p=PORT
```
```
docker run --rm -it -p PORT:PORT --gpus all ladonq/mot:0.0.1 python3 mot_server.py -p=PORT --lpr 1 --roi 0/1
```
```
docker run --rm -it -p PORT:PORT --gpus all ladonq/mot:0.0.1 python3 mot_server.py -p=PORT --fr 1 --roi 0/1
```

Use the mot_client.py script for sending video data and receiving processed data over LAN. Enter local IP nad port of MOT client docker container.
Also provide a path for video file if the path is not provided use a webcam.

```
python3 mot_client.py --ip MOT_CLIENT_IP -p MOT_CLIENT_PORT -v PATH_TO_VIDEO
```