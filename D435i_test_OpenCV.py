#import pyrealsense2 as rs
import numpy as np
import cv2
from depth_sensor import D435i

device = D435i(convert_to_color_map = True, width = 640, height = 480, enable_depth = True, enable_rgb = True, enable_infrared = True)

while True:
    avbl, depth, rgb, r_depth, ir = device.get_frame()

    if not avbl:
        device.release()
        break

    cv2.imshow('Depth', depth)
    cv2.imshow('Color', rgb)
    cv2.imshow('Raw depth', r_depth*255)
    cv2.imshow('Infrared', ir)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        device.release()
        cv2.destroyAllWindows()
        break