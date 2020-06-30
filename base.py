"""
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
"""
import getopt
import sys

import numpy as np
import cv2

from openvino.inference_engine import IECore
# from openvino.inference_engine import IENetwork, IEPlugin

import cProfile


def parse(argv: str) -> dict:
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


def main(argv):
    opts = parse(argv)
    # index = [i for i, v in enumerate(opts) if v[0][1] in 'civ'][0]

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

    # read model IR
    net = ie.read_network(model='/home/openvino/face/models/intel/face-detection-0106/FP32/face-detection-0106.xml',
                          weights='/home/openvino/face/models/intel/face-detection-0106/FP32/face-detection-0106.bin')

    # I/O blobs
    # (aux) net.inputs -> dict of DataPtr obj
    # (aux) described in model - different types of input sources for NN
    # (aux) get the keyword of first element in dict
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # pre process image input
    n, c, h, w = net.inputs[input_blob].shape

    # Load model to plugin
    exec_net = ie.load_network(network=net, device_name=opts['device'][1])

    # main app loop
    while True:
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

        if frame.shape[:-1] != (h, w):
            # log.warning("Image {} is resized from {} to {}".format(args.input[i], frame.shape[:-1], (h, w)))
            trans_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            # cv2.imshow("resized image", frame)
        # Change data layout from HWC to CHW
        trans_frame = trans_frame.transpose((2, 0, 1))  # (determined by net.inputs['image'].layout)

        # images[i] = frame
        # log.info("Batch size is {}".format(n))

        # perform inference
        # ininntialize imput layer to picture values
        # extract values form last layer
        # get rid of single element dimensions from n-D output
        # res = np.squeeze(exec_net.infer(inputs={input_blob: trans_frame})[out_blob])

        res = exec_net.infer(inputs={input_blob: trans_frame})
        res = res[out_blob]
        res = np.squeeze(res)

        # mark face
        # todo properly redo it
        # Change data layout from CHW back to HWC
        # frame = frame.transpose((1, 2, 0))
        i = 0
        while res[i][0] != -1:
            if res[i][2] > 0.5:
                pt1 = (int(round(frame.shape[:-1][1] * res[i][3])), int(round(frame.shape[:-1][0] * res[i][4])))
                pt2 = (int(round(frame.shape[:-1][1] * res[i][5])), int(round(frame.shape[:-1][0] * res[i][6])))
                rgb = (0, 255, 0)
                img = cv2.imread('/home/k16/Pictures/lenna.png')
                # cv2.rectangle(frame, pt1, pt2, rgb, thickness=3)
                frame = cv2.rectangle(img=frame, pt1=pt1, pt2=pt2, color=rgb, thickness=2)
                pos = (int(round(frame.shape[:-1][1] * res[i][3])), int(round(frame.shape[:-1][0] * res[i][6]) + 10))
                frame = cv2.putText(frame, '{}%'.format(int(round(res[i][2] * 100))),
                                    pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255, 255), 1)
            i += 1
        cv2.imshow('face', frame)

        # supported_layers = ie.query_network(net, "CPU")
        # not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # layers = net.layers
        # for layer, device in supported_layers.items():
        #     layers[layer].affinity = device

    return


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main(sys.argv[1:])
