import cv2
from typing import Dict, List
import numpy as np
import copy
from network_model import NetworkModel
from openvino.inference_engine import IECore, IENetwork


class FaceLocator(NetworkModel):
    def __init__(self, model: IENetwork, detection_threshold: float, input_blob_index=0, output_blob_index=0):
        super(FaceLocator, self).__init__(model)
        self.input_blob = list(model.inputs)[input_blob_index]
        self.input_shape = model.inputs[self.input_blob].shape  # [n, c, h, w]
        self.output_blob = list(model.outputs)[output_blob_index]
        self.output_shape = model.inputs[self.output_blob].shape
        self.detection_threshold = detection_threshold

    def __prepare_input(self, frame: np.ndarray) -> np.ndarray:
        ret = copy.deepcopy(frame)  # so as not to change original frame
        ret = cv2.resize(ret, (self.input_shape[3], self.input_shape[2]))
        ret = ret.transpose((2, 0, 1))  # set correct dimension order [h, w, c] to [c, h, w]
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
        faces = np.squeeze(faces)  # ged rid of 1 elemnet dimensions in nD array ([N, 7])
        for face in faces:
            position = FaceLocator.FacePosition(face, self.input_shape)
            if position.conf < self.detection_threshold:
                break  # faces are ordered form highest to lowest confidence
            position.fit_to_frame(frame)
            res_positions.append(position)
        return res_positions

    class FacePosition:
        def __init__(self, face: List, input_shape: List):
            self.image_id = face[0]
            self.label = face[1]
            self.conf = face[2]
            self.rect = {'x_min': face[3], 'y_min': face[4], 'x_max': face[5], 'y_max': face[6]}

            self.input_shape = input_shape  # [n, c, h, w]

        def __fit_to_frame_ration(self, frame_width: int, frame_height: int):
            """
            method for face position normalized to [0..1] interval
            :param frame:
            :return:
            """
            self.rect['x_min'] = int(round(self.rect['x_min'] * frame_width))
            self.rect['y_min'] = int(round(self.rect['y_min'] * frame_height))
            self.rect['x_max'] = int(round(self.rect['x_max'] * frame_width))
            self.rect['y_max'] = int(round(self.rect['y_max'] * frame_height))

        def __fit_to_frame_abs_pos(self, frame_width: int, frame_height: int):
            """
            method form face position given by absolute values of infered frame
            (note: that frame is after resizing and is different from target frame (frame that is going to be shown))
            :param frame:
            :return:
            """
            self.rect['x_min'] = int(round(frame_width) * ((100 / self.input_shape[4] * self.rect['x_min']) / 100))
            self.rect['y_min'] = int(round(frame_height) * ((100 / self.input_shape[3] * self.rect['y_min']) / 100))
            self.rect['x_max'] = int(round(frame_width) * ((100 / self.input_shape[4] * self.rect['x_max']) / 100))
            self.rect['y_max'] = int(round(frame_height) * ((100 / self.input_shape[3] * self.rect['y_max']) / 100))

        def fit_to_frame(self, frame: np.ndarray):
            """
            :param input_shape:
            :param frame: cv2.UMat
                same frame which is going to be shown later by openCV
                expected format: [h, w, c]
            :return:
            """
            frame_width = frame.shape[-2]
            frame_height = frame.shape[-3]
            if max(self.rect.values()) > 1:
                # absolute values
                self.__fit_to_frame_abs_pos(frame_width, frame_height)
            else:
                # normalized to [0..1] interval
                self.__fit_to_frame_ration(frame_width, frame_height)

            # trim out of frame rectangles
            for key in self.rect.keys():
                if self.rect[key] < 0:
                    self.rect[key] = 0
                elif key[0] == 'x':
                    if self.rect[key] > frame_width:
                        self.rect[key] = frame_width
                else:
                    if self.rect[key] > frame_height:
                        self.rect[key] = frame_height
