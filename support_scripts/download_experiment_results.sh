#! /bin/bash

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "First argument: -openvino, -raspberry3, -raspberry4, -nvidia"
  echo "Second argument: -merge"
  exit
fi

copy_from_openvino()
{
  sshpass -p "root" scp -P 22 root@172.17.0.2:/home/openvino/face/exp_1/res_exp_1.xlsx \
               /home/k/PycharmProjects/face-detection/experiment_1/OpenVino/
}

copy_from_nvidia()
{
  sshpass -p "root" scp -P 22 root@172.18.0.22:/home/nvidia/res_exp_1.xlsx \
               /home/k/PycharmProjects/face-detection/experiment_1/Nvidia_GPU/
}

copy_from_raspberry_4()
{
  sshpass -p "root" scp -P 22 root@192.168.0.206:/root/face-detection/experiment_1/RaspberryPi/res_exp_1.xlsx \
               /home/k/PycharmProjects/face-detection/experiment_1/RaspberryPi/res_exp_1_pi_4_CPU.xlsx

  sshpass -p "root" scp -P 22 root@192.168.0.206:/root/face-detection/experiment_1/OpenVion/res_exp_1.xlsx \
               /home/k/PycharmProjects/face-detection/experiment_1/RaspberryPi/res_exp_1_pi_4_MYRIAD.xlsx
}

copy_from_raspberry_3()
{
  sshpass -p "root" scp -P 22 root@192.168.0.207:/root/face-detection/experiment_1/RaspberryPi/res_exp_1.xlsx \
             /home/k/PycharmProjects/face-detection/experiment_1/RaspberryPi/res_exp_1_pi_3_CPU.xlsx

  sshpass -p "root" scp -P 22 root@192.168.0.207:/root/face-detection/experiment_1/OpenVion/res_exp_1.xlsx \
               /home/k/PycharmProjects/face-detection/experiment_1/RaspberryPi/res_exp_1_pi_3_MYRIAD.xlsx

}

merge_all_results()
{
    python3 append_to_result.py ../experiment_1/OpenVino/res_exp_1.xlsx
    python3 append_to_result.py ../experiment_1/Nvidia_GPU/res_exp_1.xlsx
    python3 append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_1.xlsx
    python3 append_to_result.py ../experiment_1/RaspberryPi/res_exp_1_pi_2.xlsx

}

if [[ "$#" -eq "0" ]]; then
  copy_from_openvino
  copy_from_nvidia
  copy_from_raspberry_4
  copy_from_raspberry_3
  merge_all_results
else
  if [[ "$1" == "-openvino" ]]; then
      copy_from_openvino
  elif [[ "$1" == "-nvidia" ]]; then
      copy_from_nvidia
  elif [[ "$1" == "-raspberry3" ]]; then
      copy_from_raspberry_3
  elif [[ "$1" == "-raspberry4" ]]; then
      copy_from_raspberry_4
  fi
  if [[ "$#" -eq "2" && "$2" == "-merge" ]]; then
    merge_all_results
  fi
fi



