import copy

import cv2
import numpy as np
from typing import Tuple, List, Dict

from face_locator import FaceLocator
from network_model import NetworkModel
from openvino.inference_engine import IECore, IENetwork


class LandmarksLocator(NetworkModel):

    def __init__(self, model: IENetwork, input_blob_index=0, output_blob_index=0):
        super(LandmarksLocator, self).__init__(model)
        self.input_blob = list(model.inputs)[input_blob_index]
        self.input_shape = model.inputs[self.input_blob].shape  # [n, c, h, w]
        self.output_blob = list(model.outputs)[output_blob_index]
        self.output_shape = model.outputs[self.output_blob].shape

    def __prepare_input(self, frame: np.ndarray, face_locations: List['FaceLocator.FacePosition']) -> List[np.ndarray]:
        """
        Cuts frame to individual faces and converts cut regions to shape: [1x3x48x48] for network.
            An input image should be in the format [BxCxHxW]:
                B - batch size
                C - number of channels
                H - image height
                W - image width
            The expected color order is BGR.
 
        :param frame: original frame in [h, w, c] BGR format
        :param face_locations: list of FaceLocator.FacePosition objects holding all detected faces
        :return: 
        """
        frame = frame.transpose((2, 0, 1))  # set correct dimension order from [h, w, c] to [c, h, w]
        prepared_face_frames = []
        for face in face_locations:
            prepared_face_frame = np.empty(self.input_shape[1:])
            crop = np.array(frame[:,
                            face.rect['y_min']:face.rect['y_max'],
                            face.rect['x_min']:face.rect['x_max']])
            for i, chanel in enumerate(crop):
                prepared_face_frame[i] = cv2.resize(chanel, (self.input_shape[3], self.input_shape[2]))
                pass
            prepared_face_frames.append(prepared_face_frame.reshape(self.input_shape))
        return prepared_face_frames

    def sync_infer(self, frame: np.ndarray, face_locations: List[FaceLocator.FacePosition]) -> List[
            Dict[str, np.ndarray]]:
        return [super(LandmarksLocator, self).sync_infer({self.input_blob: face})
                for face in self.__prepare_input(frame, face_locations)]

    def get_landmarks(self, frame: np.ndarray, face_locations: List[FaceLocator.FacePosition]) \
            -> List['LandmarksLocator.FaceLandmarks']:
        faces_landmarks = []
        for i, face_location in enumerate(face_locations):
            coordinates = np.squeeze(self.sync_infer(frame, face_locations)[i][self.output_blob])

            face_landmarks = LandmarksLocator.FaceLandmarks(coordinates, face_location)
            face_landmarks.fit_to_frame()
            faces_landmarks.append(face_landmarks)
        return faces_landmarks

    class FaceLandmarks:
        def __init__(self, landmarks: np.ndarray, face_location: FaceLocator.FacePosition):
            self.face_location = face_location
            # classic coordinate notation (x,y)
            self.left_eye = landmarks[0:2]
            self.right_eye = landmarks[2:4]
            self.nose = landmarks[4:6]
            self.mouth = landmarks[6:10]

        def fit_to_frame(self) -> None:
            face_width = self.face_location.rect['x_max'] - self.face_location.rect['x_min']
            face_height = self.face_location.rect['y_max'] - self.face_location.rect['y_min']

            for landmark in self.get_points():
                landmark[0] = self.face_location.rect['x_min'] + int(round(face_width * landmark[0]))
                landmark[1] = self.face_location.rect['y_min'] + int(round(face_height * landmark[1]))

        def get_points(self):
            return self.left_eye, self.right_eye, self.nose, self.mouth[:2], self.mouth[-2:]
