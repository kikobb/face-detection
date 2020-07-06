import os
import cv2
import numpy as np

from typing import Dict, Union, Tuple
# from argparse import ArgumentParser, ArgumentError, Action
import argparse
from openvino.inference_engine import IECore


class ReadableFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = values
        if not os.path.isfile(path):
            raise argparse.ArgumentError(self, 'fire: \'{}\' does not exist'.format(path))
        elif not os.access(path, os.R_OK):
            raise argparse.ArgumentError(self, 'fie: \'{}\' is not a readable file'.format(path))
        else:
            print('dest: {}, values: {}'.format(self.dest, path))
            setattr(namespace, self.dest, path)


def create_argparser():
    p = argparse.ArgumentParser()
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image', action=ReadableFile)
    input_group.add_argument('-c', '--camera', const=0, nargs='?')
    input_group.add_argument('-v', '--video', action=ReadableFile)

    models = p.add_argument_group('Models')
    models.add_argument('-dm', '--detection_model', action=ReadableFile)
    models.add_argument('-lm', '--landmarks_model', action=ReadableFile)
    models.add_argument('-im', '--identification_model', action=ReadableFile)

    p.add_argument('-d', '--device', choices=['CPU', 'MYRIAD'], required=True)

    return p


def check_args(args, p: any):
    if not (args.image or args.camera or args.video):
        p.error('At least one option (--image|--camera|--video) is required.')


class IOChanel:
    image = 0
    video = 1
    camera = 2

    def __init__(self, args: Dict):
        self.i_type, self.i_source = IOChanel.input_source_converter(args)
        self.__open_i_feed()

    @classmethod
    def input_source_converter(cls, args: Dict) -> Tuple[int, Union[int, str]]:
        if args['image']:
            return cls.image, args['image']
        elif args['video']:
            return cls.video, args['video']
        # it is guaranteed that at least one option is valid
        return cls.camera, int(args['camera'])

    def __open_i_feed(self):
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
        net_face_detect = self.load_network(args['detection_model'])

    def load_network(self, model_path: str):
        return 0

def main():
    p = create_argparser()
    args = p.parse_args()
    check_args(args, p)

    io = IOChanel(vars(args))

    while True:
        io.show_frame('frame', io.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
