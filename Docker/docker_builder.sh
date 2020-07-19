#!/bin/bash

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "no arguments, -o_base, -o_dev, -r1, -r2, -nvidia"
  exit
fi

if [[ "$#" -eq "0" ]]; then
  echo "openvino all"
	docker build -t openvino:dev_base -f /home/k/PycharmProjects/face-detection/Docker/openvino_dev_base.dockerfile .
	docker build -t openvino:dev -f /home/k/PycharmProjects/face-detection/Docker/openvino_dev.dockerfile .
	exit
fi

if [[ "$#" -eq "1" ]]; then
	if [[ "$1" == "-o_base" ]]; then
	  echo "openvino base"
		docker build -t openvino:dev_base -f /home/k/PycharmProjects/face-detection/Docker/openvino_dev_base.dockerfile .
		exit
	elif [[ "$1" == "-o_dev" ]]; then
	  echo "openvino dev"
		docker build -t openvino:dev -f /home/k/PycharmProjects/face-detection/Docker/openvino_dev.dockerfile .
		exit
	elif [[ "$1" == "-r1" ]]; then
	  echo "openvino raspbian_dev_base"
		docker build -t ov_raspberry:dev_base -f /home/k/PycharmProjects/face-detection/Docker/raspbian_dev_base.dockerfile .
		exit
	elif [[ "$1" == "-r2" ]]; then
	  echo "openvino raspbian_dev"
		docker build -t ov_raspberry:dev -f /home/k/PycharmProjects/face-detection/Docker/raspbian_dev.dockerfile .
		exit 
	elif [[ "$1" == "-nvidia" ]]; then
	  echo "nvidia"
		docker build -t tf_nvidia:dev -f /home/k/PycharmProjects/face-detection/Docker/tensorflow_nvidia.dockerfile .
		exit 
	else
		echo "unsupported argument"
		exit
	fi
fi

if [[ "$#" -gt "1" ]]; then
	echo "Invalid number of arguments ($#)."
	exit
fi

# docker build -t openvino:dev_base -f ./openvino_dev_base.dockerfile .
# docker build -t openvino:dev -f ./openvino_dev.dockerfile .


#docker run --privileged -v /dev:/dev -it openvino:dev
#docker run --privileged -v /dev:/dev --device=/dev/video0:/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it openvino:dev