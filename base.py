"""
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
"""
import getopt
import sys

import cv2


def parse(argv: str) -> dict:
    shortopts = "c:i:v:"
    longopts = ""
    mutually_excluded_opt = ('civ', ())
    ret = {
        'input': [],
        'detect-mdl': '',

    }
    try:
        # check for mutually excluded options
        for group in mutually_excluded_opt:
            occurred_f = False

            for item in group:
                if '-' + item in argv:
                    if occurred_f:
                        raise getopt.GetoptError("mutually excluded parameters occurred")
                    else:
                        occurred_f = True
        # parse arguments
        opts, args = getopt.getopt(argv, shortopts, longopts)
        if len(args) != 0:
            raise getopt.GetoptError("unsupported or too many parameters")
        # for group in mutually_excluded_opt:
        #     if len([i for i, v in enumerate(opts) if v[0][1] in group]) != 1:
        #         raise getopt.GetoptError("mutually excluded parameters occurred")
    except getopt.GetoptError:
        print('todo error invalid input')
        sys.exit(2)
    return opts


def main(argv):
    opts = parse(argv)
    index = [i for i, v in enumerate(opts) if v[0][1] in 'civ'][0]

    # open input visual feed
    try:
        cap = cv2.VideoCapture(int(opts[index][1]) if opts[index][0] == '-c' else opts[index][1])
    except ValueError:
        raise NotImplementedError("invalid camera index (-c value)")

    # if not cap.isOpened():
    #     raise IOError("visual input feed failed to open")

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    ret, frame = cap.read()
    if not ret:
        raise IOError("no fame received")

    # comment
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
