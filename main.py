import cv2 as cv
from feature_matching import FeatureMatching

def main():
    capture = cv.VideoCapture(0)
    assert capture.isOpened(), "Cannot connect to camera"

    capture.set(cv.CAP_PROP_FPS, 10)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    matching = FeatureMatching(train_image='train.png')

    for success, frame in iter(capture.read, (False, None)):
        cv.imshow("frame", frame)
        match_succsess, img_warped, img_flann = matching.match(frame)

        if match_succsess:
            cv.imshow("res", img_warped)
            cv.imshow("flann", img_flann)
        if cv.waitKey(1) & 0xff == 27:
            break
