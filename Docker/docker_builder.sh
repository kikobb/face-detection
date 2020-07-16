#!/bin/bash

if [[ "$#" -eq "0" ]]; then
	docker build -t openvino:dev_base -f ./openvino_dev_base.dockerfile .
	docker build -t openvino:dev -f ./openvino_dev.dockerfile .
	exit
fi

if [[ "$#" -eq "1" ]]; then
	if [[ "$1" -eq "1" ]]; then
		docker build -t openvino:dev_base -f ./openvino_dev_base.dockerfile .
		exit
	elif [[ "$1" -eq "2" ]]; then
		docker build -t openvino:dev -f ./openvino_dev.dockerfile .
		exit
	elif [[ "$1" -eq "nvidia" ]]; then
		docker build -t tf_nvidia:dev -f ./tensorflow_nvidia.dockerfile .
		exit 
	else
		echo "3th stage doe not exist."
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