#! /bin/bash

if [ "$#" -gt "2" ]; then
        echo "Invalid number of arguments ($#)."
        exit 1
fi

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "-a, -o, -pi4, -pi3"
  exit
fi

case $1 in
    # OpenVino app
    -o)
        sshpass -p "root" scp -r -P 22 /home/k/PycharmProjects/face-detection/model_library/* \
                                root@172.17.0.2:/home/openvino/face/models/
        ;;
    -pi4)
        sshpass -p "root" scp -r -P 22 /home/k/PycharmProjects/face-detection/model_library \
                                root@192.168.0.206:/root/face-detection/
        ;;
    -pi3)
        sshpass -p "root" scp -r -P 22 /home/k/PycharmProjects/face-detection/model_library \
                                root@192.168.0.207:/root/face-detection/
        ;;
esac