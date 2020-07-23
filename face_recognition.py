import os
import time

import cv2
import numpy as np

from typing import Dict, Union, Tuple, List
from enum import Enum
import argparse
from openvino.inference_engine import IECore, IENetwork

from face_locator import FaceLocator
from face_recognizer import FaceRecognizer
from landmarks_locator import LandmarksLocator
from measure_time import MeasureTime


class EndOfStream(Exception):
    def __init__(self, message):
        super().__init__(message)


class ReadableFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        path = values[0]

        # if model path check for weights too
        if self.container.title == 'Models':
            path_w = os.path.splitext(path)[0] + ".bin"
            if not os.path.isfile(path_w):
                raise argparse.ArgumentError(self, 'file: \'{}\' does not exist'.format(path_w))
            elif not os.access(path_w, os.R_OK):
                raise argparse.ArgumentError(self, 'file: \'{}\' is not a readable file'.format(path_w))

        if not os.path.isfile(path):
            raise argparse.ArgumentError(self, 'file: \'{}\' does not exist'.format(path))
        elif not os.access(path, os.R_OK):
            raise argparse.ArgumentError(self, 'file: \'{}\' is not a readable file'.format(path))
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
    output_group.add_argument('-on', '--output_none', action='store_true')

    models = p.add_argument_group('Models')
    models.add_argument('-dm', '--detection_model', action=ReadableFile, nargs=1, required=True)
    models.add_argument('-dm_t', '--detection_model_threshold', metavar='[0..1]', type=float, default=0.5, nargs=1)
    models.add_argument('-lm', '--landmarks_model', action=ReadableFile, nargs=1)
    models.add_argument('-rm', '--recognition_model', action=ReadableFile, nargs=1)

    p.add_argument('-d', '--device', choices=['CPU', 'MYRIAD', 'GPU'], required=True, nargs=1)
    p.add_argument('-t', '--time', action='store_true')

    return p


def check_args(args, p: any):
    # todo check for combined max model size NCS 500MB (320MB)
    if not (args.input_image or args.input_camera or args.input_video):
        p.error('At least one option ( --input_image| --input_camera| --input_video) is required.')
    if not (args.output_display or args.output_file or args.output_none):
        p.error('At least one option ( --output_display| --output_file | --output_none) is required.')


class IOChanel:
    class Input(Enum):
        IMAGE = 0
        VIDEO = 1
        CAMERA = 2

    class Output(Enum):
        DISPLAY = 0
        FILE = 1
        NONE = 2

    def __init__(self, args: Dict):
        self.i_chanel, self.i_source = IOChanel.get_input_chanel_type(args)
        self.o_chanel = IOChanel.get_output_chanel_type(args)
        self.i_feed = None 
        self.__open_i_feed()

    @classmethod
    def get_input_chanel_type(cls, args: Dict) -> Tuple['IOChanel.Input', Union[int, str]]:
        if args['input_image']:
            return cls.Input.IMAGE, args['input_image']
        elif args['input_video']:
            return cls.Input.VIDEO, args['input_video']
        # it is guaranteed that at least one option is valid
        return cls.Input.CAMERA, int(args['input_camera'])

    @classmethod
    def get_output_chanel_type(cls, args: Dict) -> 'IOChanel.Output':
        if args['output_display']:
            return cls.Output.DISPLAY
        if args['output_none']:
            return cls.Output.NONE
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
        elif self.i_chanel == self.Input.VIDEO:
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
            if self.i_chanel.VIDEO:
                raise EndOfStream('app ended successfully')
            raise IOError("no fame received")
        return frame

    # def process_frame(self, frame: np.ndarray):
    #
    #     while self.i_feed.isOpened():
    #         frame = self.get_frame().get()

    @staticmethod
    def draw_findings(frame: np.ndarray, findings: List[Union[List[FaceLocator.FacePosition],
                                                              List[LandmarksLocator.FaceLandmarks],
                                                              List[FaceRecognizer.FaceIdentity]]]) -> np.ndarray:
        rec_color = (0, 255, 0)

        # draw rectangle around detected faces, write confidence in % below
        for face, face_id in zip(findings[0], findings[2]):
            frame = cv2.rectangle(img=frame,
                                  pt1=(face.rect['x_min'], face.rect['y_min']),
                                  pt2=(face.rect['x_max'], face.rect['y_max']),
                                  color=rec_color, thickness=2)
            frame = cv2.putText(frame, '{}%'.format(int(round(face.conf * 100))),
                                (face.rect['x_min'], face.rect['y_max'] + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
            # draw label
            frame = cv2.putText(frame, '{}'.format(face_id.face_id),
                                (face.rect['x_min'], face.rect['y_min'] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)

        # draw little circle for every landmark detected
        for face_landmarks in findings[1]:
            for landmark in face_landmarks.get_points():
                cv2.circle(frame, tuple(landmark), 2, (0, 0, 255), cv2.FILLED, cv2.LINE_8)

        # todo add loop for landmarks and detection
        return frame

    def write_output(self, frame: np.ndarray):
        if self.o_chanel == self.Output.DISPLAY:
            # self.show_frame('face_recognition', frame)
            self.show_frame('face_recognition', cv2.resize(frame, (1080, 720)))
        elif self.o_chanel == self.Output.NONE:
            pass
        else:
            # todo implement other output sources
            raise NotImplementedError('other output sources than camera not implemented')

    @staticmethod
    def show_frame(name: str, frame: np.ndarray) -> None:
        cv2.imshow(name, frame)

    def __del__(self):
        self.i_feed.release()


class ProcessFrame:

    def __init__(self, args: Dict):
        # todo proper handling of self.modes
        self.ie = IECore()
        self.modes = self.__determine_processing_mode(args)
        net_face_detect = net_landmarks_detect = net_recognize_face = None

        if not self.modes['detect']:
            raise ValueError('detection model undefined')
        # load networks from file
        net_face_detect = self.__prepare_network(args['detection_model'])
        # put it to corresponding class
        self.face_locator = FaceLocator(net_face_detect, args['detection_model_threshold'])
        # setup device plugins
        if next(iter(args['device'])) == 'CPU':
            # CPU
            self.ie.set_config(config={
                "CPU_THROUGHPUT_STREAMS": "1",
                "CPU_THREADS_NUM": "8",
            }, device_name='CPU')
        elif next(iter(args['device'])) == 'GPU':
            # GPU
            self.ie.set_config(config={"GPU_THROUGHPUT_STREAMS": "1"}, device_name='GPU')
        elif next(iter(args['device'])) == 'MYRIAD':
            pass
        # load to device for inferencing
        self.face_locator.deploy_network(next(iter(args['device'])), self.ie)

        if self.modes['landmark']:
            net_landmarks_detect = self.__prepare_network(args['landmarks_model'])
            self.landmarks_locator = LandmarksLocator(net_landmarks_detect)
            self.landmarks_locator.deploy_network(next(iter(args['device'])), self.ie)

        if self.modes['recognize']:
            net_recognize_face = self.__prepare_network(args['recognition_model'])
            self.face_recognizer = FaceRecognizer(net_recognize_face)
            self.face_recognizer.deploy_network(next(iter(args['device'])), self.ie)

        # todo other models or load separately

    @staticmethod
    def __determine_processing_mode(args: Dict) -> Dict[str, bool]:
        ret = {}
        if args['detection_model']:
            ret['detect'] = True
        else:
            ret['detect'] = False

        if args['landmarks_model']:
            ret['landmark'] = True
        else:
            ret['landmark'] = False

        if args['recognition_model']:
            ret['recognize'] = True
        else:
            ret['recognize'] = False
        return ret

    def __prepare_network(self, model_path: str) -> IENetwork:
        model_path = os.path.abspath(model_path)
        model = self.ie.read_network(model=model_path, weights=os.path.splitext(model_path)[0] + ".bin")
        return model

    def process_frame(self, frame: np.ndarray) -> List[Union[List[FaceLocator.FacePosition],
                                                             List[LandmarksLocator.FaceLandmarks],
                                                             List[
                                                                 FaceRecognizer.FaceIdentity]]]:  # todo uniton with None
        faces_landmarks = faces_identities = None
        face_positions = self.face_locator.get_face_positions(frame)
        if self.modes['landmark']:
            faces_landmarks = self.landmarks_locator.get_landmarks(frame, face_positions)
        if self.modes['recognize']:
            faces_identities = self.face_recognizer.get_identities(frame, face_positions, faces_landmarks)
        return [face_positions, faces_landmarks, faces_identities]


def main():
    p = create_argparser()
    args = p.parse_args()
    check_args(args, p)

    io = IOChanel(vars(args))
    # proc = ProcessFrame(vars(args))


    # cap = cv2.VideoCapture(args.input_video)
    print(f'{type(args.input_video)}')
    # cap = cv2.VideoCapture('./test_videos/face_1/face_1_240p.mp4')

    # while(io.i_feed.isOpened()):
    #     time.sleep(0.001)
    #     _, frame = io.i_feed.read()
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     continue


    # timer = None
    # if args.time:
    #     timer = MeasureTime()

    while io.i_feed.isOpened():
        io.show_frame('frame', io.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

        try:
            if args.time:
                timer.start()
            frame = io.get_frame()
            findings = proc.process_frame(frame)
            frame = io.draw_findings(frame, findings)
            io.write_output(frame)
            if io.o_chanel == io.Output.DISPLAY and cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if args.time:
                timer.stop()
        except EndOfStream:
            break
        except IOError:
            break

    # if args.time:
    #     timer.print_data()

    io.i_feed.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
