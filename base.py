"""
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
"""
import getopt
import sys

import cv2


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
        if not next(i for i in opts if i[0][1] in opt_groups[2]) in valid_opt_formats['device']:
            raise getopt.GetoptError("unsupported format of -m")
    except getopt.GetoptError:
        print('todo error invalid input')
        sys.exit(2)
    # fill return dict

    return {
        # input chanel (photo, video, camera)
        'input': next(i for i in opts if i[0][1] in opt_groups[0]),
        # Intermediate Representation of cnn model for face localization
        'localize-mdl': next(i for i in opts if i[0][1] in opt_groups[1]),
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

    ret, frame = cap.read()
    if not ret:
        raise IOError("no fame received")

    # main app loop
    while True:
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
    return


if __name__ == '__main__':
    main(sys.argv[1:])
