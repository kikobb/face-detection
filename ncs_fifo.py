import getopt
import sys
import os

import numpy as np
import cv2
import time
from typing import Dict

from openvino.inference_engine import IECore


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


def main(argv):
    opts = parse(argv)


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main(sys.argv[1:])
