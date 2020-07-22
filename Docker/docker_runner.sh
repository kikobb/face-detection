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

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "First argument: -openvino, -raspberry, -nvidia"
  echo "Second argument: -it"
  exit
fi

run_openvino_interactive()
{
  xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			--device=/dev/video0:/dev/video0 \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-e DISPLAY=$DISPLAY\
			-it openvino:dev
	xhost -
}

run_openvino_not_interactive()
{
  xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			--device=/dev/video0:/dev/video0 \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			-e DISPLAY=$DISPLAY\
		  -di openvino:dev
	xhost -
}

run_raspberry_interactive()
{
	xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			--device=/dev/video0:/dev/video0 \
			-e DISPLAY=$DISPLAY \
			-it ov_raspberry:dev_base
	xhost -
}

run_raspberry_not_interactive()
{
	xhost +
	docker run --privileged \
			--rm \
			-v /dev:/dev \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			--device=/dev/video0:/dev/video0 \
			-e DISPLAY=$DISPLAY \
			-di ov_raspberry:dev_base
	xhost -
}

run_nvidia_interactive()
{
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
}

run_nvidia_not_interactive()
{
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
			-di tf_nvidia:dev

	xhost -
}

#run OpenVino container interactively
if [[ "$#" -eq "2" && "$1" == "-openvino" && "$2" == "-it" ]]; then
	run_openvino_interactive
	exit
fi

#run OpenVino container not interactively
if [[ "$#" -eq "1" && "$1" == "-openvino"  ]]; then
	run_openvino_not_interactive
	exit
fi

#run container on raspberry pi interactively
if [[ "$#" -eq "2" && "$1" == "-raspberry" && "$2" == "-it" ]]; then
  run_raspberry_interactive
	exit
fi

#run container on raspberry pi not interactively
if [[ "$#" -eq "1" && "$1" == "-raspberry" ]]; then
  run_raspberry_not_interactive
	exit
fi

#run tf container with nvidia support interactively
if [[ "$#" -eq "1" && "$1" == "-nvidia" ]]; then
	run_nvidia_not_interactive
	exit
fi

#run tf container with nvidia support not interactively
if [[ "$#" -eq "2" && "$1" == "-nvidia" && "$2" == "-it" ]]; then
	run_nvidia_interactive
	exit
fi

#too many args
if [[ "$#" -gt "2" ]]; then
	echo "Invalid number of arguments ($#)."
	exit
fi

echo "no or wrong arguments"

# docker run --entrypoint -v /home/k/PycharmProjects/face-detection:/opt/project --rm