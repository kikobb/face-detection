import cv2
import numpy as np

from typing import Dict, List

from face_locator import FaceLocator
from landmarks_locator import LandmarksLocator
from network_model import NetworkModel
from openvino.inference_engine import IECore, IENetwork


class FaceRecognizer(NetworkModel):
    def __init__(self, model: IENetwork, input_blob_index=0, output_blob_index=0):
        super(FaceRecognizer, self).__init__(model)
        self.input_blob = list(model.inputs)[input_blob_index]
        self.input_shape = model.inputs[self.input_blob].shape  # [n, c, h, w]
        self.output_blob = list(model.outputs)[output_blob_index]
        self.output_shape = model.outputs[self.output_blob].shape
        # self.detection_threshold = detection_threshold

    def __prepare_input(self, frame: np.ndarray,
                        face_locations: List[FaceLocator.FacePosition],
                        face_landmarks: List[LandmarksLocator.FaceLandmarks]
                        ) -> List[np.ndarray]:
        """
        Base on face_location cuts face out and base on face_landmarks transforms and normalize it
        Name: "data" , shape: [1x3x128x128] - An input image in the format [BxCxHxW], where:
            B - batch size
            C - number of channels
            H - image height
            W - image width
        Expected color order is BGR.
        :param frame: original frame in [h, w, c] BGR format
        :param face_locations:
        :param face_landmarks:
        :return:
        """
        # todo redo to actual face normalization
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

    def sync_infer(self, frame: np.ndarray,
                   face_locations: List[FaceLocator.FacePosition],
                   face_landmarks: List[LandmarksLocator.FaceLandmarks],
                   ) -> List[Dict[str, np.ndarray]]:
        return [super(FaceRecognizer, self).sync_infer({self.input_blob: face})
                for face in self.__prepare_input(frame, face_locations, face_landmarks)]

    def get_identities(self, frame: np.ndarray,
                       face_locations: List[FaceLocator.FacePosition],
                       face_landmarks: List[LandmarksLocator.FaceLandmarks],
                       ) -> List['FaceRecognizer.FaceIdentity']:
        identity_of_faces = []
        tmp = self.sync_infer(frame, face_locations, face_landmarks)
        for id in self.sync_infer(frame, face_locations, face_landmarks):
            identity = 0

        return None

    class FaceIdentity:
        def __init__(self):
            pass