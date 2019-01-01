from imutils import face_utils
from math import sqrt, atan, pi, sin, cos, fabs
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

        try:

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
            noseHeight, noseWidth, _ = nose_img.shape
            noseHeight = int(calc_euclid_distance(shape[31], shape[35]) * 2.5)
            noseWidth = noseHeight
            angle = int(atan((shape[31][1] - shape[35][1]) / (shape[35][0] - shape[31][0])) * 180 / pi)
            rot_nose_img = rotateImage(nose_img, angle)
            r_nose_img = cv2.resize(rot_nose_img, (noseWidth, noseHeight), interpolation=cv2.INTER_AREA)
            nose_image_center = tuple(np.array(r_nose_img.shape[1::-1]) / 2)
            drawing_point = 30
            nose_drawing_point = [int(shape[drawing_point][0] - nose_image_center[0]),
                                  int(shape[drawing_point][1] - nose_image_center[1])]

            overlay = cv2.addWeighted(image[nose_drawing_point[1]:(nose_drawing_point[1] + noseHeight),
                                      nose_drawing_point[0]:(nose_drawing_point[0] + noseWidth)], 1,
                                      r_nose_img, 1, 0)
            image[nose_drawing_point[1]:(nose_drawing_point[1] + noseHeight),
                  nose_drawing_point[0]:(nose_drawing_point[0] + noseWidth)] = overlay

            # draw the ears of the filter

            # find angle of eyes to the horizontal line
            angle = int(atan((shape[36][1] - shape[45][1]) / (shape[45][0] - shape[36][0])) * 180 / pi)
            # this is the offset angle to the horizontal -> if we shift this with 90 degrees it is also offset to the
            # vertical
            # from this we can conclude that the sin of angle times the length of the offset in our case 200 will gib the
            # offset of x
            xOffset = angle + 30
            yOffset = 100
            ears_drawing_point = [int(shape[36][0] - xOffset),
                                  int(shape[36][1] - yOffset)]
            rot_ears_img = rotateImage(ears_img, angle)
            earsWidth = int(calc_euclid_distance(shape[36], shape[45]) * 2)
            earsHeight = int(earsWidth * 0.5)
            r_ears_img = cv2.resize(rot_ears_img, (earsWidth, earsHeight), interpolation=cv2.INTER_AREA)
            earsHeight, earsWidth, _ = r_ears_img.shape
            overlay = cv2.addWeighted(image[ears_drawing_point[1]:(ears_drawing_point[1] + earsHeight),
                                      ears_drawing_point[0]:(ears_drawing_point[0] + earsWidth)], 1,
                                      r_ears_img, 1, 0)
            image[ears_drawing_point[1]:(ears_drawing_point[1] + earsHeight),
                  ears_drawing_point[0]:(ears_drawing_point[0] + earsWidth)] = overlay

        except:
            print("Out of bound")

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # for (x, y) in shape:
        #    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()