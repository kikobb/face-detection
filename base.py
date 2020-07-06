"""
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
"""
import getopt
import sys
import os

import numpy as np
import cv2
import time
from typing import Dict

from openvino.inference_engine import IECore
# from openvino.inference_engine import IENetwork, IEPlugin


def parse(argv: str) -> Dict:
    # order matches return dictionary order ( [0] = input, [1] = localize-mdl, [2] = device, ...)
    opt_groups = (':civ', ':m', ':d')
    valid_opt_formats = {'device': ('CPU', 'GPU', 'MYRIAD')}
    shortopts = ''
    for group in opt_groups:
        if group[0] == ':':
            shortopts += ''.join([opt + ':' for opt in group[1:]])
        else:
            shortopts += group[1:]

        # elif group[0] == '=':
        #     pass

    try:
        # parse arguments
        opts, args = getopt.getopt(argv, shortopts)
        if len(args) != 0:
            raise getopt.GetoptError("unsupported or too many parameters")
        if len(opts) == 0:
            raise getopt.GetoptError("no parameters")
        # check for mutually excluded options
        for group in opt_groups:
            count = 0
            for i in opts:
                if i[0][1] in group:
                    count += 1
            if count > 1:
                raise getopt.GetoptError("mutually excluded parameters occurred")
        # validate argument parameters
        if not next((i for i in opts if i[0][1] in opt_groups[2]), None)[1] in valid_opt_formats['device']:
            raise getopt.GetoptError("unsupported format of -m")
    except getopt.GetoptError as e:
        print(e, file=sys.stderr)
        print('todo error invalid input')
        sys.exit(2)
    # fill return dict

    return {
        # input chanel (photo, video, camera)
        'input': next((i for i in opts if i[0][1] in opt_groups[0]), None),
        # Intermediate Representation of cnn model for face localization
        'localize-model': next((i for i in opts if i[0][1] in opt_groups[1]), None),
        # hw device to perform detection
        'device': next((i for i in opts if i[0][1] in opt_groups[2]), None)
    }


def init_landmarks(ie: IECore, loc: Dict[str, str]) -> Dict:
    # 5 face landmarks detection
    landmarks_net = ie.read_network(model='{0}/{1}/FP32/{1}.xml'.format(loc['dir'], loc['name']),
                                         weights='{0}/{1}/FP32/{1}.bin'.format(loc['dir'], loc['name']))
    i_blob_name = list(landmarks_net.inputs)[0]
    # get input dimensions
    n, c, h, w = landmarks_net.inputs[i_blob_name].shape


def main(argv):
    opts = parse(argv)
    # index = [i for i, v in enumerate(opts) if v[0][1] in 'civ'][0]

    models_dir = '/home/openvino/face/models/intel'                     # directory of all models
    fce_det_v = 0                                                       # version of face detection model
    fce_det_n = 'face-detection-010{}'.format(fce_det_v)                # name of face detection model
    fce_landmarks_n = 'landmarks-regression-retail-0009'                # name of face landmarks detection model

    # open input visual feed
    try:
        cap = cv2.VideoCapture(int(opts['input'][1]) if opts['input'][0] == '-c' else opts['input'][1])
    except ValueError:
        raise NotImplementedError("invalid camera index (-c value)")
    if not cap.isOpened():
        raise IOError("visual input feed not found")

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # BGR
    ret, frame = cap.read()
    if not ret:
        raise IOError("no fame received")

    # plugin = IEPlugin(device='CPU')

    # load plugin
    ie = IECore()

    # read model IRs
    # face detection
    net_face_detect = ie.read_network(model='{0}/{1}/FP32/{1}.xml'.format(models_dir, fce_det_n),
                          weights='{0}/{1}/FP32/{1}.bin'.format(models_dir, fce_det_n))
    # init_landmarks(ie, {'dir':models_dir, 'name':fce_landmarks_n})

    # I/O blobs
    # (aux) net_face_detect.inputs -> dict of DataPtr obj
    # (aux) described in model - different types of input sources for NN
    # (aux) get the keyword of first element in dict
    input_blob = list(net_face_detect.inputs)[0]
    if fce_det_v == 0:
        out_blob = list(net_face_detect.outputs)[0]
    else:
        out_blob = list(net_face_detect.outputs)[1]

    # pre process image input
    n, c, h, w = net_face_detect.inputs[input_blob].shape

    # Load model to plugin ExecutableNetwork
    exec_net = ie.load_network(network=net_face_detect, device_name=opts['device'][1])

# TMP test
    net_face_detect2 = ie.read_network(model='{0}/face-detection-0105/FP32/face-detection-0105.xml'.format(models_dir),
                                      weights='{0}/face-detection-0105/FP32/face-detection-0105.bin'.format(models_dir))
    exec_net2 = ie.load_network(network=net_face_detect2, device_name=opts['device'][1])
    out_blob2 = list(net_face_detect2.outputs)[1]
    n2, c2, h2, w2 = net_face_detect2.inputs[input_blob].shape

# TMP load plugin timer
#     for i in range(10):
#         start_load = time.time()
#         exec_net = ie.load_network(network=net_face_detect, device_name=opts['device'][1])
#         stop_load = time.time()
#         print('load {0} duration: {1}ns'.format(i, stop_load - start_load))

    # main app loop
    while True:
        start_time = time.time()  # start time of the loop (fps)
        # ---- Read input frame
        if not cap.grab():
            raise IOError("no fame received")
        # handle keyboard input
        key_stroke = cv2.waitKey(1)
        if key_stroke != -1:
            if key_stroke == ord('s'):
                raise NotImplementedError("save current frame")
            if key_stroke == ord('q'):
                break

        # cv2.imshow('frame', frame)
        # (aux) frame matrix format: BGR todo check if desirable
        ret, frame = cap.retrieve()
        if not ret:
            raise NotImplementedError("end of visual feed")
#TMP test
        frame2 = frame.copy()
        if frame2.shape[:-1] != (h2, w2):
            # log.warning("Image {} is resized from {} to {}".format(args.input[i], frame.shape[:-1], (h, w)))
            trans_frame2 = cv2.resize(frame2, (w2, h2), interpolation=cv2.INTER_AREA)
            # cv2.imshow("resized image", frame)
        # Change data layout from HWC to CHW
        trans_frame2 = trans_frame2.transpose((2, 0, 1))  # (determined by net_face_detect.inputs['image'].layout)

        if frame.shape[:-1] != (h, w):
            # log.warning("Image {} is resized from {} to {}".format(args.input[i], frame.shape[:-1], (h, w)))
            trans_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            # cv2.imshow("resized image", frame)
        # Change data layout from HWC to CHW
        trans_frame = trans_frame.transpose((2, 0, 1))  # (determined by net_face_detect.inputs['image'].layout)

        # images[i] = frame
        # log.info("Batch size is {}".format(n))

        # perform inference
        # ininntialize imput layer to picture values
        # extract values form last layer
        # get rid of single element dimensions from n-D output
        # res = np.squeeze(exec_net.infer(inputs={input_blob: trans_frame})[out_blob])

        res = exec_net.infer(inputs={input_blob: trans_frame})
        # TMP test
        res2 = exec_net2.infer(inputs={input_blob: trans_frame2})

        rec_color = (0, 255, 0)
        #sotres cropped faces
        faces = []
        # mark face
        # todo properly redo it
        # Change data layout from CHW back to HWC
        # frame = frame.transpose((1, 2, 0))
        if fce_det_v == 0:
            res = res[out_blob]
            res = np.squeeze(res)
            i = 0
            while res[i][2] > 0.5:
                pt1 = (int(round(frame.shape[:-1][1] * res[i][3])), int(round(frame.shape[:-1][0] * res[i][4])))
                pt2 = (int(round(frame.shape[:-1][1] * res[i][5])), int(round(frame.shape[:-1][0] * res[i][6])))
                faces.append(frame[pt1[1]:pt2[1], pt1[0]:pt2[0]])
                # faces.append(cv2.resize(frame,))
                # cv2.rectangle(frame, pt1, pt2, rgb, thickness=3)
                frame = cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=rec_color, thickness=2)
                frame = cv2.putText(frame, '{}%'.format(int(round(res[i][2] * 100))),
                                    (pt1[0], pt2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
                i += 1
            #TMP test
            j = 0
            while res2[out_blob2][j][4] > 0.5:
                pt1 = (
                    int(round(frame2.shape[:-1][1] * ((100 / w2 * res2[out_blob2][j][0]) / 100))),
                    int(round(frame2.shape[:-1][0] * ((100 / h2 * res2[out_blob2][j][1]) / 100))))
                pt2 = (
                    int(round(frame2.shape[:-1][1] * ((100 / w2 * res2[out_blob2][j][2]) / 100))),
                    int(round(frame2.shape[:-1][0] * ((100 / h2 * res2[out_blob2][j][3]) / 100))))
                # cv2.rectangle(frame, pt1, pt2, rgb, thickness=3)
                frame2 = cv2.rectangle(img=frame2, pt1=pt1, pt2=pt2, color=rec_color, thickness=2)
                frame2 = cv2.putText(frame2, '{}%'.format(int(round(res2[out_blob2][j][4] * 100))),
                                    (pt1[0], pt2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)

                j += 1
        elif fce_det_v >= 5:
            i = 0
            while res['labels'][i] != -1:
                if res[out_blob][i][4] > 0.5:
                    pt1 = (
                        int(round(frame.shape[:-1][1] * ((100 / w * res[out_blob][i][0]) / 100))),
                        int(round(frame.shape[:-1][0] * ((100 / h * res[out_blob][i][1]) / 100))))
                    pt2 = (
                        int(round(frame.shape[:-1][1] * ((100 / w * res[out_blob][i][2]) / 100))),
                        int(round(frame.shape[:-1][0] * ((100 / h * res[out_blob][i][3]) / 100))))
                    # cv2.rectangle(frame, pt1, pt2, rgb, thickness=3)
                    frame = cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=rec_color, thickness=2)
                    frame = cv2.putText(frame, '{}%'.format(int(round(res[out_blob][i][4] * 100))),
                                        (pt1[0], pt2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)

                i += 1
        elif fce_det_v == 6:
            i = 0
            # while res['labels'][i] != -1:
            #     if res[out_blob][i][4] > 0.5:
            #         pt1 = (int(round(res[out_blob][i][0])), int(round(frame.shape[:-1][0] * (100 / h * res[out_blob][i][1]))))
            #         pt2 = (int(round(res[out_blob][i][2])), int(round(frame.shape[:-1][0] * (100 / h * res[out_blob][i][3]))))
            #         img = cv2.imread('/home/k16/Pictures/lenna.png')
            #         # cv2.rectangle(frame, pt1, pt2, rgb, thickness=3)
            #         frame = cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=rec_color, thickness=2)
            #         frame = cv2.putText(frame, '{}%'.format(int(round(res[out_blob][i][4] * 100))),
            #                             (pt1[0], pt2[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
            #     i += 1
        else:
            raise IOError("model not found") # TODO: change classification of this exception

        frame = cv2.putText(frame, 'FPS: {:.2f}'.format(1.0 / (time.time() - start_time)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
        cv2.imshow('frame', frame)
        #TMP test
        frame2 = cv2.putText(frame2, 'FPS: {:.2f}'.format(1.0 / (time.time() - start_time)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
        cv2.imshow('frame2', frame2)
        # faces = np.asarray(faces)
        #corpp image for landmarks detection
        i = 0
        for face in faces:
            if face.size !=0: cv2.imshow('face{}'.format(i), face)
            i += 1

    return


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main(sys.argv[1:])
