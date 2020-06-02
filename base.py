"""
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
"""
import getopt
import sys

import cv2
from openvino.inference_engine import IECore
# from openvino.inference_engine import IENetwork, IEPlugin


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
        # check for mutually excluded options
        for group in opt_groups:
            count = 0
            for i in opts:
                if i[0][1] in group:
                    count += 1
            if count > 1:
                raise getopt.GetoptError("mutually excluded parameters occurred")
        # validate argument parameters
        if not next(i for i in opts if i[0][1] in opt_groups[2])[1] in valid_opt_formats['device']:
            raise getopt.GetoptError("unsupported format of -m")
    except getopt.GetoptError as e:
        print(e, file=sys.stderr)
        print('todo error invalid input')
        sys.exit(2)
    # fill return dict

    return {
        # input chanel (photo, video, camera)
        'input': next(i for i in opts if i[0][1] in opt_groups[0]),
        # Intermediate Representation of cnn model for face localization
        'localize-model': next(i for i in opts if i[0][1] in opt_groups[1]),
        # hw device to perform detection
        'device': next(i for i in opts if i[0][1] in opt_groups[2])
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

        cv2.imshow('frame', frame)

        ret, frame = cap.retrieve()
        if not ret:
            raise NotImplementedError("end of visual feed")

        # ---- Load plugins for inference engine
        # plugins_for_devices = {'': 0}  # todo change variable
        # plugins_for_devices = {str: }  # todo change variable
        # cmd_options = [(opts['device'], opts['localize-model'])]  # todo may be redundant
        #
        # for option in cmd_options:
        #     device_name = option[0]     # todo may be redundant
        #     network_name = option[0]    # todo may be redundant
        #
        #     if device_name == '' or network_name == '':
        #         continue
        #     if

        # plugin = IEPlugin(device='CPU')

        # load plugin
        ie = IECore()

        # read model IR
        net = ie.read_network(model='/home/k16/vboxshared/face-detection-0100.xml',
                              weights='/home/k16/vboxshared/face-detection-0100.bin')

        # I/O blobs
        input_blob = next(iter(net.inputs))
        out_blob = next(iter(net.outputs))


        # Load model to plugin
        exec_net = ie.load_network(network=net, device_name='CPU')

        # pre process image input
        n, c, h, w = net.inputs[input_blob].shape
        # images = np.ndarray(shape=(n, c, h, w))
        # for i in range(n):

        if frame.shape[:-1] != (h, w):
            # log.warning("Image {} is resized from {} to {}".format(args.input[i], frame.shape[:-1], (h, w)))
            frame = cv2.resize(frame, (w, h))
        frame = frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW

            # images[i] = frame
        # log.info("Batch size is {}".format(n))

        # performe inference
        res = exec_net.infer(inputs={input_blob: frame})

        print(res)

        # supported_layers = ie.query_network(net, "CPU")
        # not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        # layers = net.layers
        # for layer, device in supported_layers.items():
        #     layers[layer].affinity = device

    return


if __name__ == '__main__':
    main(sys.argv[1:])
