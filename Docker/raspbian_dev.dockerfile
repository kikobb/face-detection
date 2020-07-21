FROM ov_raspberry:dev_base

USER root

#####################
# REQUIRED SOFTWARE #
RUN apt-get update && \
      apt-get -y install sudo 
RUN apt-get -y install vim
RUN sudo apt-get -y install ssh

# install missing libraryes
RUN apt-get -y install python3-pip
RUN pip3 install numpy==1.15.4 Pillow openpyxl
RUN apt-get -y install libsm6

#################
#   USERS & SSH #
#setup ssh connection
RUN echo "" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd
# # add apssword to user openvino
# RUN echo "openvino:openvino" | chpasswd