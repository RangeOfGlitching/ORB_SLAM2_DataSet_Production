from create_camera import *
rs = IntelRealSense()
counter = 1
try:
    while True:
        ret, color_image, depth_image = rs.get_frame_stream()
        if not ret:
            continue
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        rs.save_pic(color_image, depth_image_3d, counter)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_3d, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((depth_colormap, color_image))
        key = cv2.waitKey(1)
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        counter += 1
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    rs.pipeline.stop()
