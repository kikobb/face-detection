import cv2
import numpy as np
from typing import Dict, List
import copy
from abc import ABC, abstractmethod

from custom_exceptions import UnknownNetworkType
from network_identifier import NetworkType
from network_model import NetworkModel
from openvino.inference_engine import IECore, IENetwork


class FaceLocator(NetworkModel):
    def __init__(self, model: IENetwork, detection_threshold: float, network_type: NetworkType):
        super(FaceLocator, self).__init__(model)
        self.network_type = network_type
        self.input_blob = list(model.inputs)[network_type.get_input_blob_index()]
        self.input_shape = model.inputs[self.input_blob].shape  # [n, c, h, w]
        self.output_blob = list(model.outputs)[network_type.get_output_blob_index()]
        self.output_shape = model.outputs[self.output_blob].shape
        self.detection_threshold = detection_threshold

    def __prepare_input(self, frame: np.ndarray) -> np.ndarray:
        ret = copy.deepcopy(frame)  # so as not to change original frame
        ret = cv2.resize(ret, (self.input_shape[3], self.input_shape[2]))
        ret = ret.transpose((2, 0, 1))  # set correct dimension order from [h, w, c] to [c, h, w]
        ret = ret.reshape(self.input_shape)
        return ret

    def sync_infer(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        return super(FaceLocator, self).sync_infer({self.input_blob: self.__prepare_input(frame)})

    def get_face_positions(self, frame: np.ndarray) -> List['FaceLocator.FacePosition']:
        # from documentation:
        # The net outputs blob with shape: [1, 1, N, 7], where N is the number of detected bounding boxes.
        # For each detection, the description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        res_positions = []
        faces = self.sync_infer(frame)[self.output_blob]
        faces = np.squeeze(faces)  # ged rid of 1 element dimensions in nD array ([N, 7])
        for face in faces:
            position = FaceLocator.FacePosition.get_instance(self.network_type, face, self.input_shape)
            if position.conf < self.detection_threshold:
                break  # faces are ordered form highest to lowest confidence
            position.fit_to_frame(frame)
            res_positions.append(position)
        return res_positions

    class FacePosition(ABC):
        @staticmethod
        def get_instance(network_type: NetworkType, face: List, input_shape: List):
            if network_type.get_name() == 'face-detection-0100' or network_type.get_name() == 'face-detection-0102' or \
                    network_type.get_name() == 'face-detection-0104':
                return FaceLocator.FacePosition0100(face, input_shape)
            elif network_type.get_name() == 'face-detection-0105' or network_type.get_name() == 'face-detection-0106':
                return FaceLocator.FacePosition0105(face, input_shape)
            else:
                raise UnknownNetworkType(f"Network {network_type.get_name()} has no adequate FacePosition class")

        def __init__(self, confidence: float, face_coordinates: List, input_shape: List):
            self.conf = confidence
            self.rect = {
                'x_min': face_coordinates[0],
                'y_min': face_coordinates[1],
                'x_max': face_coordinates[2],
                'y_max': face_coordinates[3]
            }
            self.input_shape = input_shape  # [n, c, h, w]

        def _fit_to_frame_ratio(self, frame_width: int, frame_height: int):
            """
            method for face position normalized to [0..1] interval
            :param frame_width:
            :param frame_height:
            :return:
            """
            self.rect['x_min'] = int(round(self.rect['x_min'] * frame_width))
            self.rect['y_min'] = int(round(self.rect['y_min'] * frame_height))
            self.rect['x_max'] = int(round(self.rect['x_max'] * frame_width))
            self.rect['y_max'] = int(round(self.rect['y_max'] * frame_height))

        def _fit_to_frame_abs_pos(self, frame_width: int, frame_height: int):
            """
            method form face position given by absolute values of infered frame
            (note: that frame is after resizing and is different from target frame (frame that is going to be shown))
            :param frame_width:
            :param frame_height:
            :return:
            """

            self.rect['x_min'] = int(round(frame_width) * ((100 / self.input_shape[3] * self.rect['x_min']) / 100))
            self.rect['y_min'] = int(round(frame_height) * ((100 / self.input_shape[2] * self.rect['y_min']) / 100))
            self.rect['x_max'] = int(round(frame_width) * ((100 / self.input_shape[3] * self.rect['x_max']) / 100))
            self.rect['y_max'] = int(round(frame_height) * ((100 / self.input_shape[2] * self.rect['y_max']) / 100))

        def _rect_trim_to_frame(self, frame_width: int, frame_height: int):
            """
            trim rectangles that are out of frame
            :param frame_width:
            :param frame_height:
            :return:
            """
            for key in self.rect.keys():
                if self.rect[key] < 0:
                    self.rect[key] = 0
                elif key[0] == 'x':
                    if self.rect[key] > frame_width:
                        self.rect[key] = frame_width
                else:
                    if self.rect[key] > frame_height:
                        self.rect[key] = frame_height

        @abstractmethod
        def fit_to_frame(self, frame: np.ndarray):
            pass

    class FacePosition0105(FacePosition):
        def __init__(self, face: List, input_shape: List):
            super(FaceLocator.FacePosition0105, self).__init__(face[-1], face[0:-1], input_shape)

        def fit_to_frame(self, frame: np.ndarray):
            """
            :param frame: same frame which is going to be shown later by openCV expected format: [h, w, c]
            :return:
            """
            frame_width = frame.shape[-2]
            frame_height = frame.shape[-3]
            super(FaceLocator.FacePosition0105, self)._fit_to_frame_abs_pos(frame_width, frame_height)
            super(FaceLocator.FacePosition0105, self)._rect_trim_to_frame(frame_width, frame_height)

    class FacePosition0100(FacePosition):
        def __init__(self, face: List, input_shape: List):
            super(FaceLocator.FacePosition0100, self).__init__(face[2], face[3:7], input_shape)
            self.image_id = face[0]
            self.label = face[1]

        def fit_to_frame(self, frame: np.ndarray):
            """
            :param frame: same frame which is going to be shown later by openCV expected format: [h, w, c]
            :return:
            """
            frame_width = frame.shape[-2]
            frame_height = frame.shape[-3]
            # values should be normalize to [0..1] sometimes could overshoot by negligible margin
            # normalized to [0..1] interval
            super(FaceLocator.FacePosition0100, self)._fit_to_frame_ratio(frame_width, frame_height)
            super(FaceLocator.FacePosition0100, self)._rect_trim_to_frame(frame_width, frame_height)
