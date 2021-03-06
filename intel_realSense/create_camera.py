import pyrealsense2 as rs
import numpy as np
import cv2
import time


class IntelRealSense:
    def __init__(self):
        self.file = open("fr1_xyz.txt", "w", encoding="UTF-8")
        self.pipeline = rs.pipeline()
        config = rs.config()
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        self.clipping_distance_in_meters = 1  # 1 meter
        clipping_distance = self.clipping_distance_in_meters / self.depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame_stream(self):
        alignFlag = True
        # Get frames of color and depth
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            alignFlag = False

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return alignFlag, color_image, depth_image

    def save_pic(self, colorImage, depthImage_3d, counter):
        # use timeStamp as file's name
        addressRgb = rf"rgb/{counter}.PNG"
        addressDepth = rf"depth/{counter}.PNG"
        cv2.imwrite(addressRgb, colorImage)
        cv2.imwrite(addressDepth, depthImage_3d)

        self.file.write(f"{counter} {addressRgb} {counter} {addressDepth}\n")

    def release(self):
        self.pipeline.stop()
        self.file.close()


if __name__ == '__main__':
    pass
