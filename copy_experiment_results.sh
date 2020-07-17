#! /bin/bash

sshpass -p "root" scp -P 22 root@172.17.0.2:/home/openvino/face/exp_1/res_exp_1.xlsx \
             /home/k/PycharmProjects/face-detection/experiment_1/OpenVino/

sshpass -p "root" scp -P 22 root@172.18.0.22:/home/nvidia/res_exp_1.xlsx \
             /home/k/PycharmProjects/face-detection/experiment_1/Nvidia_GPU/

python3 merge_two_exp.py experiment_1/OpenVino/res_exp_1.xlsx experiment_1/Nvidia_GPU/res_exp_1.xlsx compl_exp_1.xlsx
