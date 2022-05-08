import socket, cv2, pickle, struct
import numpy as np

client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = '172.17.176.1' # paste your server ip address here
port = 8000
client_socket.connect((host_ip,port)) # a tuple

data = b""
payload_size = struct.calcsize("Q")


use_sensor = True
send_depth_map = False
video_file_path = None
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
        continue


    if send_depth_map:
        cv2.imshow('Object tracking',np.hstack((depth, frame)))
    else:
        cv2.imshow('Object tracking',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        client_socket.close()
        cv2.destroyAllWindows()
        break