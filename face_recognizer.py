import cv2
import numpy as np
from scipy.spatial.distance import cosine

from typing import Dict, List, Tuple

# from face_library import FaceLibrary
from face_locator import FaceLocator
from landmarks_locator import LandmarksLocator
from network_model import NetworkModel
from openvino.inference_engine import IECore, IENetwork


class FaceRecognizer(NetworkModel):
    def __init__(self, model: IENetwork, face_id_threshold=0.3, input_blob_index=0, output_blob_index=0):
        super(FaceRecognizer, self).__init__(model)
        self.input_blob = list(model.inputs)[input_blob_index]
        self.input_shape = model.inputs[self.input_blob].shape  # [n, c, h, w]
        self.output_blob = list(model.outputs)[output_blob_index]
        self.output_shape = model.outputs[self.output_blob].shape
        self.face_library = FaceLibrary()
        self.face_id_threshold = face_id_threshold

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
        # tmp = self.sync_infer(frame, face_locations, face_landmarks)
        for face_descriptor in self.sync_infer(frame, face_locations, face_landmarks):
            face_descriptor = np.squeeze(face_descriptor[self.output_blob])
            face_id, dist = self.face_library.identify_face(face_descriptor)
            if dist > self.face_id_threshold:
                face_id = 'Unknown'
            identity_of_faces.append(FaceRecognizer.FaceIdentity(face_id, dist))
        return identity_of_faces

    class FaceIdentity:
        def __init__(self, face_id: str, dist: float):
            self.face_id = face_id
            self.dist = dist


def cosine_dist(x, y):
    return cosine(x, y) * 0.5


class FaceLibrary:
    def __init__(self):
        self.me = self.load_face()

    def identify_face(self, face_descriptor: np.ndarray) -> Tuple[str, float]:
        return 'Kristian', cosine_dist(face_descriptor, self.me)

    def save_face(self, face_descriptor: np.ndarray) -> None:
        np.savetxt('kristian_face_descriptor.csv', face_descriptor, fmt='%f', delimiter=',')

    def load_face(self) -> np.ndarray:
        return np.loadtxt('kristian_face_descriptor.csv', delimiter=',')
