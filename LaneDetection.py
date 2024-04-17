import numpy as np
import cv2


def draw_lines(image, lines):
    # create a distinct image for the lines [0,255] - all the 0 values are black
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are x, y points for starting and end points for linex
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)

    # finally we have to merge the  image  with lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines


def region_of_interest(image, region_points):
    # We are replacing pixels with 0 (black) - the region we are not interested
    mask = np.zeros_like(image)

    # the region we are interested in the lower triangle - replace with 255
    cv2.fillPoly(mask, region_points, 255)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # convert color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny s algorithm)
    canny_image = cv2.Canny(gray_image, 100, 200)

    # we are interested about the lower region of the image (There are the driving lanes)
    region_of_interested_vertices = [
        (0, height),
        (width / 2, height * 0.65),
        (width, height)
    ]

    # we can get rid of the un-relevant part of the image
    # We just keep the lower triangle region
    cropped_image = region_of_interest(canny_image, np.array([region_of_interested_vertices], np.int32))

    # use line detection algorithm. (radians instead of degrees. 1 degree = pi/180)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40,
                            maxLineGap=150)

    # draw the lines with image
    image_with_lines = draw_lines(image, lines)

    return image_with_lines


# video = several frame of pictures
video = cv2.VideoCapture('videos/lane_detection_video.mp4')

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
