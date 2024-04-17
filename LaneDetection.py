import numpy as np
import cv2

# video = several frame of pictures

video = cv2.VideoCapture('videos/lane_detection_video.mp4')

while video.isOpened():

    # it returns two item. frame and video status
    is_grabbed, frame = video.read()

    # if video is ended

    if not is_grabbed:
        break

    cv2.imshow("Lane detection video", frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
