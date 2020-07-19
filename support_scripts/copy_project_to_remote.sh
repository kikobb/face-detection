#! /bin/bash

if [ "$#" -gt "2" ]; then
        echo "Invalid number of arguments ($#)."
        exit 1
fi

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "-a, -e1o, -e1g, -rasp4, -rasp3"
  exit
fi

case $1 in
    # OpenVino app
    -a)
        for file in $(find /home/k/PycharmProjects/face-detection/ -maxdepth 1 -type f ! -name "*.xlsx"); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face
        done
        exit 0
        ;;
    # OpenVino experiment 1
    -e1o)
        for file in $(find /home/k/PycharmProjects/face-detection/experiment_1/OpenVino -maxdepth 1 -type f ! -name "*.xlsx"); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face/exp_1
        done
        for file in $(find /home/k/PycharmProjects/face-detection/experiment_1/ -maxdepth 1 -type f ! -name "*.xlsx"); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face/exp_1
        done
        exit 0
	      ;;
    # GPU Nvidia experiment 1
    -e1g)
        for file in $(find /home/k/PycharmProjects/face-detection/experiment_1/Nvidia_GPU -maxdepth 1 -type f ! -name "*.xlsx"); do
            sshpass -p "root" scp -P 22 "$file" root@172.18.0.22:/home/nvidia
        done
        for file in $(find /home/k/PycharmProjects/face-detection/experiment_1/ -maxdepth 1 -type f ! -name "*.xlsx"); do
            sshpass -p "root" scp -P 22 "$file" root@172.18.0.22:/home/nvidia
        done
        exit 0
        ;;
    -rasp4)
      sshpass -p "root" ssh root@192.168.0.206 'cd ./face-detection && git pull'
      exit 0
      ;;
    -rasp3)
      sshpass -p "root" ssh root@192.168.0.207 'cd ./face-detection && git pull'
      exit 0
      ;;
esac


	
