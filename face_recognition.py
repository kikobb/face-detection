import os
import cv2
import numpy as np

from typing import Dict, Union, Tuple, List
from enum import Enum
import argparse
from openvino.inference_engine import IECore, IENetwork

from face_locator import FaceLocator


class ReadableFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = values[0]

        # if model path check for weights too
        if self.container.title == 'Models':
            path_w = os.path.splitext(path)[0] + ".bin"
            if not os.path.isfile(path_w):
                raise argparse.ArgumentError(self, 'fire: \'{}\' does not exist'.format(path_w))
            elif not os.access(path_w, os.R_OK):
                raise argparse.ArgumentError(self, 'fie: \'{}\' is not a readable file'.format(path_w))

        if not os.path.isfile(path):
            raise argparse.ArgumentError(self, 'fire: \'{}\' does not exist'.format(path))
        elif not os.access(path, os.R_OK):
            raise argparse.ArgumentError(self, 'fie: \'{}\' is not a readable file'.format(path))
        else:
            setattr(namespace, self.dest, path)


def create_argparser():
    p = argparse.ArgumentParser()
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument('-ii', '--input_image', action=ReadableFile, nargs=1)
    input_group.add_argument('-ic', '--input_camera', const=0, nargs='?')
    input_group.add_argument('-iv', '--input_video', action=ReadableFile, nargs=1)

    output_group = p.add_mutually_exclusive_group()  # todo check if file is possible to write
    output_group.add_argument('-od', '--output_display', action='store_true')
    output_group.add_argument('-of', '--output_file', nargs=1)

    models = p.add_argument_group('Models')
    models.add_argument('-dm', '--detection_model', action=ReadableFile, nargs=1)
    models.add_argument('-dm_t', '--detection_model_threshold', metavar='[0..1]', type=float, default=0.5, nargs=1)
    models.add_argument('-lm', '--landmarks_model', action=ReadableFile, nargs=1)
    models.add_argument('-im', '--identification_model', action=ReadableFile, nargs=1)

    p.add_argument('-d', '--device', choices=['CPU', 'MYRIAD'], required=True, nargs=1)

    return p


def check_args(args, p: any):
    # todo check for combined max model size NCS 500MB (320MB)
    if not (args.input_image or args.input_camera or args.input_video):
        p.error('At least one option ( --input_image| --input_camera| --input_video) is required.')
    if not (args.output_display or args.output_file):
        p.error('At least one option ( --output_display| --output_file) is required.')


class IOChanel:
    class Input(Enum):
        IMAGE = 0
        VIDEO = 1
        CAMERA = 2

    class Output(Enum):
        DISPLAY = 0
        FILE = 1

    def __init__(self, args: Dict):
        self.i_chanel, self.i_source = IOChanel.input_chanel_converter(args)
        self.o_chanel = IOChanel.output_chanel_converter(args)
        self.i_feed = None
        self.__open_i_feed()

    @classmethod
    def input_chanel_converter(cls, args: Dict) -> Tuple['IOChanel.Input', Union[int, str]]:
        if args['input_image']:
            return cls.Input.IMAGE, args['input_image']
        elif args['input_video']:
            return cls.Input.VIDEO, args['input_video']
        # it is guaranteed that at least one option is valid
        return cls.Input.CAMERA, int(args['input_camera'])

    @classmethod
    def output_chanel_converter(cls, args: Dict) -> 'IOChanel.Output':
        if args['output_display']:
            return cls.Output.DISPLAY
        # it is guaranteed that at least one option is valid
        return cls.Output.FILE

    def __open_i_feed(self) -> None:
        if self.i_chanel == self.Input.CAMERA:
            try:
                self.i_feed = cv2.VideoCapture(self.i_source)
            except ValueError:
                raise NotImplementedError("invalid camera index (-c value)")
            if not self.i_feed.isOpened():
                raise IOError("visual input feed not found")
        else:
            # todo implement other input sources
            raise NotImplementedError('other input sources than camera not implemented')

    def get_frame(self) -> np.ndarray:
        received, frame = self.i_feed.read()
        if not received:
            raise IOError("no fame received")
        return frame

    # def process_frame(self, frame: np.ndarray):
    #
    #     while self.i_feed.isOpened():
    #         frame = self.get_frame().get()

    @staticmethod
    def draw_findings(frame: np.ndarray, findings: List[Union[List['FaceLocator.FacePosition'], None, None]]) -> np.ndarray:
        rec_color = (0, 255, 0)

        for face in findings[0]:
            frame = cv2.rectangle(img=frame,
                                  pt1=(face.rect['x_min'], face.rect['y_min']),
                                  pt2=(face.rect['x_max'], face.rect['y_max']),
                                  color=rec_color, thickness=2)
            frame = cv2.putText(frame, '{}%'.format(int(round(face.conf * 100))),
                                (face.rect['x_min'], face.rect['y_max'] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
        # todo add loop for landmarks and detection
        return frame

    def write_output(self, frame: np.ndarray):
        if self.o_chanel == self.Output.DISPLAY:
            self.show_frame('face_recognition', frame)
        else:
            # todo implement other output sources
            raise NotImplementedError('other output sources than camera not implemented')

    @staticmethod
    def show_frame(name: str, frame: np.ndarray) -> None:
        cv2.imshow(name, frame)


class ProcessFrame:

    def __init__(self, args: Dict):
        self.ie = IECore()
        net_face_detect = self.__prepare_network(args['detection_model'])
        # todo other models

        self.face_locator = FaceLocator(net_face_detect, args['detection_model_threshold'])
        # todo other models

        self.face_locator.deploy_network(next(iter(args['device'])), self.ie)  # load network to device
        # todo other models or load separately

    def __prepare_network(self, model_path: str) -> IENetwork:
        model_path = os.path.abspath(model_path)
        model = self.ie.read_network(model=model_path, weights=os.path.splitext(model_path)[0] + ".bin")
        return model

    def process_frame(self, frame: np.ndarray) -> List[Union[List['FaceLocator.FacePosition'], None, None]]:
        face_positions = self.face_locator.get_face_positions(frame)
        return [face_positions, None, None]


def main():
    p = create_argparser()
    args = p.parse_args()
    check_args(args, p)

    io = IOChanel(vars(args))
    proc = ProcessFrame(vars(args))

    while True:
        # io.show_frame('frame', io.get_frame())
        frame = io.get_frame()
        findings = proc.process_frame(frame)
        frame = io.draw_findings(frame, findings)
        io.write_output(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
