# YOLOv4 multi object tracking (MOT) with DeepSORT

YOLOv4 and DeepSORT object tracking application in automatic license plate recognition and face recognition. Face recognition system also uses face depth maps, 
extracted using D435i sensor, as a complementary verification to rgb images.

## Usage with Docker

### MOT docker container setup

First create the MOT docker container. You can stop it by running exit or docker stop commands.

```
docker run --name mot -it -p PORT:PORT --gpus all ladonq/mot:0.0.1
```

After the container was created the following python scripts can be run inside the container after starting it in interactive mode.

Below you can see the mot_server.py script commands with different arguments. Enter a port to use default is 9999 
and whether to activate license plate recognition (--lpr) or face recognition (--fr). The region of interests can be activated (--roi) 
to select a part of a screen.

```
docker start -i mot

python3 mot_server.py -p=PORT

python3 mot_server.py -p=PORT --lpr 1 --roi 0/1

python3 mot_server.py -p=PORT --fr 1 --roi 0/1
```

Use the mot_client.py script for sending video data and receiving processed data over LAN. Enter local IP (or use localhost) nad port of MOT server docker container.
Also provide a path for video file, if the path is not provided the webcam will be used.

```
python3 mot_client.py --ip localhost -p PORT -v PATH_TO_VIDEO
```

### PostgreSQL database setup for face recognition

To save images for face recognition create a PostgreSQL database docker container.

```
docker run --name postgresql -e POSTGRES_PASSWORD=docker -d postgres
```

Find the IP address of postgresql container in the docker bridge virtual network with this command.

```
docker network inspect bridge
```

As mentioned earlier start the MOT container if not running and run once the pgsql_script.py script with the postgresql container IP address as an argument to create the face database.

```
docker start -i mot

python3 pgsql_script.py --db_hostname POSTGRESQL_CONTAINER_IP_ADDRESS
```

Finally to save the face embeddings to the database run the get_face_embeddings.py script where n_img means the number of images taken per person (default is 10). Also run the mot_client.py script with the collect_data flag.

```
docker start -i mot

python3 pgsql_script.py --db_hostname POSTGRESQL_CONTAINER_IP_ADDRESS

python3 get_face_embeddings.py --n_img 10 --port PORT --db_hostname POSTGRESQL_CONTAINER_IP_ADDRESS

python mot_client.py --ip localhost -p PORT -v PATH_TO_VIDEO --collect_data 1
```