FROM tensorflow/tensorflow:latest-gpu-py3  

USER root

RUN apt-get update && \
      apt-get -y install sudo 
RUN apt-get -y install vim
RUN sudo apt-get -y install ssh
RUN apt-get -y install x11-xserver-utils

# install missing libraryes
RUN apt-get -y install python3-pip
RUN pip3 install numpy Pillow matplotlib


#setup ssh connection
RUN echo "" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd

################################
# COPY AND CREATE FIEL STRUCT. #
#project direcorty
ENV PROJECT_DIR=/home/nvidia
ENV MODEL_DIR=$PROJECT_DIR/models
ENV MOBILENET_V2_DIR=mobilenet_v2
RUN mkdir $PROJECT_DIR

# import mobilenet_v2 checkpoints and convert to IR
WORKDIR $MODEL_DIR/tmp_mobilenet
# RUN mkdir $MODEL_DIR
RUN mkdir ../$MOBILENET_V2_DIR
COPY /$MOBILENET_V2_DIR/mobilenet*.tgz $MODEL_DIR/tmp_mobilenet/
# unpac all networks
RUN for file in $(ls | grep .tgz); do \
mkdir ../$MOBILENET_V2_DIR/${file%.*}; \
tar -xzf $file -C ../$MOBILENET_V2_DIR/${file%.*}; \
done; 
WORKDIR $MODEL_DIR/$MOBILENET_V2_DIR
RUN rm -rf ../tmp_mobilenet

#sert up entrypoint script
COPY nvidia_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/nvidia_entrypoint.sh
# backwards compat
RUN ln -s /usr/local/bin/nvidia_entrypoint.sh / 
ENTRYPOINT ["nvidia_entrypoint.sh"]


WORKDIR $PROJECT_DIR
CMD ["/bin/bash"]