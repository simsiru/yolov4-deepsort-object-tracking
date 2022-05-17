import socket, cv2, pickle, struct
import numpy as np
import argparse


def stream_video_data_over_lan(host_ip, port = 9999, use_sensor = True,
send_depth_map = False, video_file_path = None, data_collect_mode = False):
    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client_socket.connect((host_ip,port))

    data = b""
    payload_size = struct.calcsize("Q")


    if use_sensor:
        from depth_sensor import D435i
        sensor = D435i(convert_to_color_map = True, width = 640, height = 480,
        enable_depth = True, enable_rgb = True, enable_infrared = False)
        dm_xmin, dm_ymin, dm_xmax, dm_ymax = 180, 80, 460, 400
    else:
        if video_file_path is not None:
            vid = cv2.VideoCapture(video_file_path)
        else:
            vid = cv2.VideoCapture(0)

    client_command = "0"

    while True:
        if use_sensor:
            avbl, depth, frame, raw_depth, _ = sensor.get_frame()
            if not avbl:
                sensor.release()
                break
            if send_depth_map:
                cv2.rectangle(depth, (dm_xmin, dm_ymax), (dm_xmax, dm_ymin), (255,255,255), 2)
        else:
            ret, frame = vid.read()
            if not ret:
                break


        a = pickle.dumps(frame)
        message = struct.pack("Q",len(a))+a
        client_socket.sendall(message)

        if send_depth_map:
            a = pickle.dumps(raw_depth)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)

        if data_collect_mode:
            a = pickle.dumps(client_command)
            message = struct.pack("Q",len(a))+a
            client_socket.sendall(message)
            #client_socket.sendall(bytes(client_command, "utf-8"))
            client_command = "0"


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
        except Exception:
            pass


        if send_depth_map:
            cv2.imshow('Object tracking',np.hstack((depth, frame)))
        else:
            cv2.imshow('Object tracking',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            client_socket.close()
            cv2.destroyAllWindows()
            break



        if cv2.waitKey(1) & 0xFF == ord("a") and data_collect_mode:
            client_command = "a"

        if cv2.waitKey(1) & 0xFF == ord("s") and data_collect_mode:
            client_command = "s"

        if cv2.waitKey(1) & 0xFF == ord("d") and data_collect_mode:
            client_command = "d"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start webcam or depth sensor stream to YOLOv4 MOT over LAN')
    parser.add_argument('--ip', type=str, required=True,
    help='Local IP of MOT docker container')
    parser.add_argument('-p', '--port', type=int, default=9999,
    help='Port for MOT docker container')
    parser.add_argument('-s', '--sensor', type=bool, default=False,
    help='Whether or not to use a realsense D435i sensor')
    parser.add_argument('--dm', type=bool, default=False,
    help='Whether or not to use face depth maps when face recognition and sensor is active')
    parser.add_argument('-v', '--video_file_path', default=None,
    help='Use a video for tracking and if the path is not provided use a webcam')
    parser.add_argument('--collect_data', type=bool, default=False,
    help='Collect face embeddings and depth maps')

    args = parser.parse_args()

    stream_video_data_over_lan(args.ip, args.port, args.sensor, args.dm,
    args.video_file_path, args.collect_data)