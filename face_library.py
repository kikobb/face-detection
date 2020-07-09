# import numpy as np
# from typing import Tuple, List, Dict
#
# from face_recognizer import FaceRecognizer
#
#
# def cosine_dist(x, y):
#     return cosine(x, y) * 0.5
#
#
# class FaceLibrary:
#     def __init__(self):
#         self.me = np.array(0)
#
#     def identify_face(self, face_descriptor: np.ndarray) -> Tuple[str, float]:
#         return 'Kristian', cosine_dist(face_descriptor, self.me)
