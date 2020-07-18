#! /bin/bash

if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "-a, -e1o, -e1g, -rasp"
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

merge_all_results()
{
    python3 merge_two_exp.py experiment_1/OpenVino/res_exp_1.xlsx experiment_1/Nvidia_GPU/res_exp_1.xlsx compl_exp_1.xlsx

}

if [[ "$#" -eq "0" ]]; then
  copy_from_openvino
  copy_from_nvidia
  merge_all_results
else
  if [[ "$1" == "openvino" ]]; then
      copy_from_openvino
  elif [[ "$1" == "nvidia" ]]; then
      copy_from_nvidia
  fi
  if [[ "$#" -eq "2" && "$2" == "merge" ]]; then
    merge_all_results
  fi
fi



