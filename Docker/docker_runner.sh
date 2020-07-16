#!/bin/bash

#HINT
#enable access to movidius NCS: 	-v /dev:/dev 
#enable access to web camera: 		--device=/dev/video0:/dev/video0 
#enable gui support for opencv: 	-e DISPLAY=$DISPLAY
#								 	-v /tmp/.X11-unix:/tmp/.X11-unix
#name the container:				--name $1
#execute in interactive mode:		-i
#specify source image:				openvino:dev

#xhost + / xhost -
#disable and enable access control for X server for all users
#TLDR: enable to show opencv gui in docker 




#run container without name
if [[ "$#" -eq "0" ]]; then
	xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			--device=/dev/video0:/dev/video0 \
			--device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl --device /dev/nvidia-uvm:/dev/nvidia-uvm\
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-e DISPLAY=$DISPLAY\
			-it openvino:dev
	xhost -
	exit
fi


#run container with name
if [[ "$#" -eq "1" && "$1" -ne "nvidia" ]]; then
	xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			--device=/dev/video0:/dev/video0 \
			-e DISPLAY=$DISPLAY \
			--name $1 \
			-it openvino:dev
	xhost -
	exit
fi

#run tf container with nvidia support
if [[ "$#" -eq "1" && "$1" -eq "nvidia" ]]; then
	NETWORK_NAME="gpu_net"
	if [[ ! $(docker network ls | tail -n +2 | awk '{print $2}' | grep $NETWORK_NAME) ]]; then
		docker network create --subnet=172.18.0.0/16 $NETWORK_NAME
	fi
	xhost +

	docker run --privileged \
			--gpus all \
			--rm \
			-v /dev:/dev \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			--device=/dev/video0:/dev/video0 \
			-e DISPLAY=$DISPLAY \
			--net $NETWORK_NAME \
			--ip 172.18.0.22 \
			-it tf_nvidia:dev

	xhost -
fi


#too many args
if [[ "$#" -gt "1" ]]; then
	echo "Invalid number of arguments ($#)."
	exit
fi


# docker run --entrypoint -v /home/k/PycharmProjects/face-detection:/opt/project --rm