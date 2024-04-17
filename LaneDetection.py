import numpy as np
import cv2


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # convert color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny s algorithm)
    canny_image = cv2.Canny(gray_image, 100, 200)

    return canny_image


# video = several frame of pictures
video = cv2.VideoCapture('videos/lane_detection_video.mp4')
# video = cv2.VideoCapture('videos/v2.mp4')

while video.isOpened():

    # it returns two item. frame and video status
    is_grabbed, frame = video.read()

    # if video is ended

    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow("Lane detection video", frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()
