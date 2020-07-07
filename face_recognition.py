import os
import cv2
import numpy as np

from typing import Dict, Union, Tuple
# from argparse import ArgumentParser, ArgumentError, Action
import argparse
from openvino.inference_engine import IECore

from face_locator import FaceLocator


class ReadableFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = values

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
    input_group.add_argument('-i', '--image', action=ReadableFile, nargs=1)
    input_group.add_argument('-c', '--camera', const=0, nargs='?')
    input_group.add_argument('-v', '--video', action=ReadableFile, nargs=1)

    models = p.add_argument_group('Models')
    models.add_argument('-dm', '--detection_model', action=ReadableFile, nargs=1)
    models.add_argument('-dm_t', '--detection_model_threshold', metavar='[0..1]', type=float, default=0.5, nargs=1)
    models.add_argument('-lm', '--landmarks_model', action=ReadableFile, nargs=1)
    models.add_argument('-im', '--identification_model', action=ReadableFile, nargs=1)

    p.add_argument('-d', '--device', choices=['CPU', 'MYRIAD'], required=True, nargs=1)

    return p


def check_args(args, p: any):
    # todo check for combined max model size NCS 500MB (320MB)
    if not (args.image or args.camera or args.video):
        p.error('At least one option (--image|--camera|--video) is required.')


class IOChanel:
    image = 0
    video = 1
    camera = 2

    def __init__(self, args: Dict):
        self.i_type, self.i_source = IOChanel.input_source_converter(args)
        self.feed = None
        self.__open_i_feed()

    @classmethod
    def input_source_converter(cls, args: Dict) -> Tuple[int, Union[int, str]]:
        if args['image']:
            return cls.image, args['image']
        elif args['video']:
            return cls.video, args['video']
        # it is guaranteed that at least one option is valid
        return cls.camera, int(args['camera'])

    def __open_i_feed(self) -> None:
        try:
            self.feed = cv2.VideoCapture(self.i_source)
        except ValueError:
            raise NotImplementedError("invalid camera index (-c value)")
        if not self.feed.isOpened():
            raise IOError("visual input feed not found")

    def get_frame(self) -> cv2.UMat:
        ret, frame = self.feed.read()
        if not ret:
            raise IOError("no fame received")
        return frame

    @staticmethod
    def show_frame(name: str, frame: cv2.UMat) -> None:
        cv2.imshow(name, frame)


class ProcessImage:

    def __init__(self, args: Dict):
        self.ie = IECore()
        net_face_detect = self.prepare_network(args['detection_model'])
        # todo other models

        self.face_locator = FaceLocator(net_face_detect, args['detection_model_threshold'])
        # todo other models

        self.face_locator.deploy_network(args['device'], self.ie)   # load network to device

    def prepare_network(self, model_path: str) -> IECore.IENetwork:
        model_path = os.path.abspath(model_path)
        model = self.ie.read_network(model=model_path, weights=os.path.splitext(model_path)[0] + ".bin")
        return model


def main():
    p = create_argparser()
    args = p.parse_args()
    check_args(args, p)

    io = IOChanel(vars(args))
    proc = ProcessImage(vars(args))

    while True:
        # io.show_frame('frame', io.get_frame())
        frame = io.get_frame()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
