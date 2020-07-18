#! /bin/bash

if [ "$#" -gt "2" ]; then
        echo "Invalid number of arguments ($#)."
        exit 1
fi

case $1 in
    # OpenVino app
    -a)
        for file in $(find . -maxdepth 1 -type f); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face
        done
        exit 0
        ;;
    # OpenVino experiment 1
    -e1o)
        for file in $(find ./experiment_1/OpenVino -maxdepth 1 -type f); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face/exp_1
        done
        for file in $(find ./experiment_1/ -maxdepth 1 -type f); do
            sshpass -p "root" scp -P 22 "$file" root@172.17.0.2:/home/openvino/face/exp_1
        done
        exit 0
	      ;;
    # GPU Nvidia experiment 1
    -e1g)
        for file in $(find ./experiment_1/Nvidia_GPU -maxdepth 1 -type f); do 
            sshpass -p "root" scp -P 22 "$file" root@172.18.0.22:/home/nvidia
        done
        for file in $(find ./experiment_1/ -maxdepth 1 -type f); do
            sshpass -p "root" scp -P 22 "$file" root@172.18.0.22:/home/nvidia
        done
        exit 0
        ;;
    -rasp)
      sshpass -p "pi" scp -P 22 "base.sh" pi@192.168.0.206:/home/pi

      for file in $(find ./experiment_1/Nvidia_GPU -maxdepth 1 -type f); do
      done
      for file in $(find ./experiment_1/ -maxdepth 1 -type f); do
          sshpass -p "root" scp -P 22 "$file" root@172.18.0.22:/home/nvidia
      done
      exit 0
      ;;
esac


	
