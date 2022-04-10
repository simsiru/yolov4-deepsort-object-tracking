import pyrealsense2 as rs
import numpy as np
import cv2

class D435i:
    def __init__(self, convert_to_color_map = True, width = 640, height = 480, enable_depth = True, enable_rgb = False, enable_infrared = False):
        """
        Realsense D435i interface

        """
        self.pipeline = rs.pipeline()
        config = rs.config()

        #pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        #pipeline_profile = config.resolve(pipeline_wrapper)
        #device = pipeline_profile.get_device()
        #device_product_line = str(device.get_info(rs.camera_info.product_line))
        self.width = width
        self.height = height

        self.enable_depth = enable_depth
        self.enable_rgb = enable_rgb
        self.enable_infrared = enable_infrared

        if not self.enable_depth and not self.enable_rgb and not self.enable_infrared:
            self.enable_depth = True

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        if self.enable_rgb:
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        if self.enable_infrared:
            config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, 30)
            config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, 30)


        self.color_map = convert_to_color_map

        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return False, None, None, None, None

            depth_image = np.asanyarray(depth_frame.get_data())

        else:
            depth_image = None
    
        if self.enable_rgb:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None, None, None, None

            color_image = np.asanyarray(color_frame.get_data())

        else:
            color_image = None

        if self.enable_infrared:
            infrared1_frame = frames.get_infrared_frame(1)
            infrared2_frame = frames.get_infrared_frame(2)
            if not infrared1_frame or not infrared2_frame:
                return False, None, None, None, None

            infrared1_image = np.asanyarray(infrared1_frame.get_data())
            infrared2_image = np.asanyarray(infrared2_frame.get_data())
        else:
            infrared1_image = None
            infrared2_image = None

        if self.color_map and self.enable_depth:
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # COLORMAP_PLASMA
            # COLORMAP_JET
            # COLORMAP_PARULA
            # COLORMAP_VIRIDIS
            # COLORMAP_CIVIDIS
            # COLORMAP_WINTER
            # COLORMAP_BONE
            # COLORMAP_TWILIGHT
            depth_map = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_PLASMA)
        else:
            depth_map = None

        return True, depth_map, color_image, depth_image, infrared1_image

    def release(self):
        self.pipeline.stop()

