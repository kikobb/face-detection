#! /bin/bash

sshpass -p "root" scp -r -P 22 root@172.17.0.2:/home/openvino/face/models/* \
               /home/k/PycharmProjects/face-detection/model_library/