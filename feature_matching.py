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

        
