from imutils import face_utils
from math import sqrt, atan, pi, degrees
import dlib
import cv2
import numpy as np


def calc_euclid_distance(first_coord, second_coord):
    return sqrt((first_coord[0] - second_coord[0]) ** 2 + (first_coord[1] - second_coord[1]) ** 2)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D((image_center[0], image_center[1]), angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

ears_img = cv2.imread('pictures/ears.png', cv2.IMREAD_COLOR)
nose_img = cv2.imread('pictures/nose.png', cv2.IMREAD_COLOR)
tongue_img = cv2.imread('pictures/tongue.png', cv2.IMREAD_COLOR)


while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    image = cv2.flip(image, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 0)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # detect the open mouth
        # calculate the distance between the 63 and 67 facial landmark
        open_mouth_distance = calc_euclid_distance(shape[63], shape[57])

        if open_mouth_distance >= 20:
            # draw the image over the ther img
            # the tongue will be as tall as the width of the mouse
            tongueHeight = int(calc_euclid_distance(shape[49], shape[55]))
            tongueWidth = tongueHeight
            angle = int(atan((shape[48][1] - shape[64][1]) / (shape[64][0] - shape[48][0])) * 180 / pi)
            rot_tongue_img = rotateImage(tongue_img, angle)
            r_tongue_img = cv2.resize(rot_tongue_img, (tongueHeight, tongueWidth), interpolation=cv2.INTER_AREA)
            drawing_point = 60
            overlay = cv2.addWeighted(image[shape[drawing_point][1]:shape[drawing_point][1] + tongueWidth,
                                            shape[drawing_point][0]:shape[drawing_point][0] + tongueHeight], 1,
                                      r_tongue_img, 1, 0)
            image[shape[drawing_point][1]:(shape[drawing_point][1] + tongueWidth),
                  shape[drawing_point][0]:(shape[drawing_point][0] + tongueHeight)] = overlay

        # draw the nose of the filter
        noseHeight = int(calc_euclid_distance(shape[49], shape[55]))
        noseWidth = int(calc_euclid_distance(shape[49], shape[55]))

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()