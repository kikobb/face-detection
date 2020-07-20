#!/bin/bash

# default is empty to use default defined in .py files
INFERENCE_REQUESTS=''

if [[ "$#" -eq "1" ]]; then
    INFERENCE_REQUESTS="$1"
fi

#check if docker is running
if [[ "$(systemctl is-active docker)" != "active" ]]; then
  echo "docker service is not running"
  exit 1
fi

# check if is needed to create images
# !!! base images has to be first in list (they are base for following images)
NECESSARY_IMAGES='openvino:dev_base openvino:dev tf_nvidia:dev'
AVAILABLE_IMAGES="$(docker images | awk -F ' ' '{print $1":"$2}' | tail -n +2)"

for IMAGE in $NECESSARY_IMAGES; do
  if ! echo "$AVAILABLE_IMAGES" | grep ^"$IMAGE"$; then
    case $IMAGE in
      openvino:dev)
        echo "- docker_builder"
        /bin/bash ../Docker/docker_builder.sh -o_dev
        ssh-keygen -f "/home/k/.ssh/known_hosts" -R "172.17.0.2"
        echo "+ done docker_builder"
        ;;
      openvino:dev_base)
        /bin/bash ../Docker/docker_builder.sh -o_base
        ;;
      tf_nvidia:dev)
        /bin/bash ../Docker/docker_builder.sh -nvidia
        ssh-keygen -f "/home/k/.ssh/known_hosts" -R "172.18.0.22"
        ;;
    esac
  fi
done

# run necessary docker images one by one and perform experimentd_1 consecutively
# to mitigate any performance throttles

# OpenVino on PC: CPU, Intel GPU, MYRIAD
echo '- docker container openvino'
/bin/bash ../Docker/docker_runner.sh -openvino
#accepts ssh key and set up enviroment variables
ssh-keygen -f "/home/k/.ssh/known_hosts" -R "172.17.0.2"
sleep 0.5
sshpass -p "root" ssh -o "StrictHostKeyChecking no" root@172.17.0.2 ":"
echo '+ done docker container openvino'
echo '- copy_project_to_remote openvino'
/bin/bash ../support_scripts/copy_project_to_remote.sh -e1o
echo '+ done copy_project_to_remote openvino'
echo '- upload_model_library openvino'
/bin/bash ../support_scripts/upload_model_library.sh -o
echo '+ done upload_model_library openvino'
echo '- experiment_1_OpenVino.py openvino'
sshpass -p "root" ssh root@172.17.0.2 "source /opt/intel/openvino/bin/setupvars.sh; python3 /home/openvino/face/exp_1/experiment_1_OpenVino.py $INFERENCE_REQUESTS"
echo '+ done experiment_1_OpenVino.py openvino'
echo '- download_experiment_results openvino'
/bin/bash ../support_scripts/download_experiment_results.sh -openvino
echo '+ done download_experiment_results openvino'
echo '- append_to_result openvino'
python3 ../support_scripts/append_to_result.py /home/k/PycharmProjects/face-detection/experiment_1/OpenVino/res_exp_1.xlsx
echo '+ done append_to_result openvino'
echo '- docker kill openvino'
docker kill "$(docker container ls --last 1 -q)"
echo '+ done docker kill openvino'

sleep 10

# Nvidia on PC: GPU
/bin/bash ../Docker/docker_runner.sh -nvidia
#accepts ssh key
ssh-keygen -f "/home/k/.ssh/known_hosts" -R "172.18.0.22"
sleep 0.5
sshpass -p "root" ssh -o "StrictHostKeyChecking no" root@172.18.0.22 ":"
/bin/bash ../support_scripts/copy_project_to_remote.sh -e1g
# necessary models are uploaded during build (dockerfile)
sshpass -p "root" ssh root@172.18.0.22 "python3 /home/nvidia/experiment_1_nvidia.py $INFERENCE_REQUESTS"
/bin/bash ../support_scripts/download_experiment_results.sh -nvidia
python3 ../support_scripts/append_to_result.py ../experiment_1/Nvidia_GPU/res_exp_1.xlsx
docker kill "$(docker container ls --last 1 -q)"



# Raspbery Pi4: CPU, MYRIAD
/bin/bash ../support_scripts/copy_project_to_remote.sh -rasp4
/bin/bash ../support_scripts/upload_model_library.sh -pi4
# experiment with OenVino library on Myriad
ssh-keygen -f "/home/k/.ssh/known_hosts" -R "192.168.0.206"
sleep 0.5
sshpass -p "root" ssh -o "StrictHostKeyChecking no" root@192.168.0.206 ":"
# experiment with tensorflow lite on CPU
sshpass -p "root" ssh root@192.168.0.206 "python3 /root/face-detection/experiment_1/RaspberryPi/experiment_1_raspberry.py $INFERENCE_REQUESTS"
read -p "Waiting to insert MYRIAD usb-stick for next test. Press enter to continue."
sshpass -p "root" ssh root@192.168.0.206 "source /opt/intel/openvino/bin/setupvars.sh; python3 /root/face-detection/experiment_1/OpenVino/experiment_1_OpenVino.py $INFERENCE_REQUESTS raspberry"
/bin/bash ../support_scripts/download_experiment_results.sh -raspberry4
python3 ../support_scripts/append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_4_CPU.xlsx
python3 ../support_scripts/append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_4_MYRIAD.xlsx

# Raspbery Pi3: CPU, MYRIAD
/bin/bash ../support_scripts/copy_project_to_remote.sh -rasp3
/bin/bash ../support_scripts/upload_model_library.sh -pi3
ssh-keygen -f "/home/k/.ssh/known_hosts" -R "192.168.0.207"
sleep 0.5
sshpass -p "root" ssh -o "StrictHostKeyChecking no" root@192.168.0.207 ":"
# experiment with tensorflow lite on CPU
sshpass -p "root" ssh root@192.168.0.207 "python3 /root/face-detection/experiment_1/RaspberryPi/experiment_1_raspberry.py $INFERENCE_REQUESTS"
# experiment with OenVino library on Myriad
#read -p "Waiting to insert MYRIAD usb-stick for next test. Press enter to continue."
sshpass -p "root" ssh root@192.168.0.207 "python3 /root/face-detection/experiment_1/OpenVino/experiment_1_OpenVino.py $INFERENCE_REQUESTS raspberry"
/bin/bash ../support_scripts/download_experiment_results.sh -raspberry3
python3 ../support_scripts/append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_3_MYRIAD.xlsx
python3 ../support_scripts/append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_3_CPU.xlsx

exit 0
