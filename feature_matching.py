import numpy as np
import cv2 as cv
from typing import Tuple, Optional, List, Sequence
Point = Tuple[float, float]

class FeatureMatching:
    def __init__(self, train_image: str = "train.png") -> None:
        self.f_extractor = cv.xfeatures2d_SURF.create(hessianThreshold=400)
        self.img_obj = cv.imread(train_image, cv.CV_8UC1)
        assert self.img_obj is not None, f"Could not find train image {train_image}"

        self.sh_train = self.img_obj.shape[:2]

        self.key_train, self.desc_train = self.f_extractor.detectAndCompute(self.img_obj, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(index_params, search_params)

        self.last_hinv = np.zeros((3,3))
        self.max_error_hinv = 50.
        self.num_frames_no_success = 0
        self.max_frames_no_success = 5

    def match(self, frame: np.ndarray) -> Tuple[bool,
                                                Optional[np.ndarray],
                                                Optional[np.ndarray]]:
        img_query = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sh_query = img_query.shape

        #key_query = self.f_extractor.detect(img_query)
        #key_query, desc_query = self.f_extractor.compute(img_query, key_query)
        key_query, desc_query = self.f_extractor.detectAndCompute(img_query, None)
        good_matches = self.match_features(desc_query)
        train_points = [self.key_train[good_match.queryIdx].pt
                        for good_match in good_matches]
        query_points = [key_query[good_match.trainIdx].pt
                        for good_match in good_matches]

    def match_features(self, desc_frame: np.ndarray) -> List[cv.DMatch]:
        matches = self.flann.knnMatch(self.desc_train, desc_frame, k=2)
        good_matches = [x[0] for x in matches
                        if x[0].distance < 0.7 * x[1].distance]
        return good_matches

def detect_corner_points(src_points: Sequence[Point],
                         dst_points: Sequence[Point],
                         sh_src: Tuple[int, int]) -> np.ndarray:
    H, _ = cv.findHomography(np.array(src_points), np.array(dst_points),
                                      cv.RANSAC)
    if H is None:
        raise Outlier("Homography not fount")
    height, width = sh_src
    src_corners = np.array([(0,0), (width, 0),
                            (width, height),
                            (0, height)], dtype=np.float32)
    return cv.perspectiveTransform(src_corners[None, :, :], H)[0]

def draw_good_matches(self, img1: np.ndarray,
                      kp1: Sequence(cv.KeyPoint),
                      img2: np.ndarray,
                      kp2: Sequence(cv.KeyPoint),
                      matches: Sequence[cv.DMatch]) -> np.ndarray:
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')

    out[:rows1, :cols1, :] = img1[..., None]
    out[:rows2, cols1: cols1 + cols2, :] = img2[..., None]

    for m in matches:
        c1 = tuple(map(int, kp1[m.queryIdx].pt))
        c2 = tuple(map(int, kp2[m.trainIdx].pt))
        c2 = c2[0] + cols1, c2[1]

        radius = 4
        BLUE = (255, 0, 0)
        thickness = 1

        cv.circle(out, c1, radius, BLUE, thickness)
        cv.circle(out, c2, radius, BLUE, thickness)

        cv.line(out, c1, c2, BLUE, thickness)
    return out



