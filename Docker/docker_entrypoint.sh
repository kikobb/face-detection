#!/bin/bash

service ssh start

# source /opt/intel/openvino/bin/setupvars.sh 

# python3 base.py
# export XAUTHORITY=~/.Xauthority

# enble to open gui windows 
xhost +

#disables Gtk-Messag / Warning / Error messages
export NO_AT_BRIDGE=1

/bin/bash