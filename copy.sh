#! /bin/bash

for file in $(find . -maxdepth 1 -type f); do 
	sshpass -p "root" scp -P 22 $file root@172.17.0.2:/home/openvino/face
done
	