#!/bin/bash

service ssh start
source /opt/intel/openvino/bin/setupvars.sh 

# python3 base.py
# export XAUTHORITY=~/.Xauthority

# enble to open gui windows 
xhost +

#disables Gtk-Messag / Warning / Error messages
export NO_AT_BRIDGE=1

# modify console prompt
cat >  /root/.bashrc << EOF

PS1='${debian_chroot:+($debian_chroot)}\u@:\W\$ '
EOF

# copy models from model_library
# if [[ "$DOWNLOAD_MODELS" == "False" ]]; then
# 	./usr/local/bin/downoad_model_library.sh
# fi

/bin/bash