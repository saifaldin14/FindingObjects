import numpy as np
import cv2
from typing import Tuple, Optional, List, Sequence
Point = Tuple[float, float]

class FeatureMatching:
    def __init__(self, train_image: str = "train.png") -> None:
        self.f_extractor =