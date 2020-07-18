#! /bin/bash

if [ "$#" -gt "2" ]; then
        echo "Invalid number of arguments ($#)."
        exit 1
fi

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "-a, -o, -pi"
  exit
fi

case $1 in
    # OpenVino app
    -o)
        sshpass -p "root" scp -r -P 22 /home/k/PycharmProjects/face-detection/model_library/* \
                                root@172.17.0.2:/home/openvino/face/models/
        ;;
    -pi)
        sshpass -p "pi" scp -r -P 22 /home/k/PycharmProjects/face-detection/model_library \
                                pi@192.168.0.206:/home/pi/openvino/face-detection/
        ;;
esac