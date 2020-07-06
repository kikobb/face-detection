FROM openvino:dev_base

USER root

RUN apt-get update && \
      apt-get -y install sudo 
RUN apt-get -y install vim
RUN sudo apt-get -y install ssh

# install missing libraryes
RUN apt-get -y install python3-pip
RUN pip3 install numpy==1.15.4
RUN apt-get -y install libsm6

#prepare enviroment (compile lib, set paths, ...)
WORKDIR /opt/intel/openvino/install_dependencies/
RUN chmod +x install_openvino_dependencies.sh
RUN ./install_openvino_dependencies.sh
WORKDIR /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/
RUN chmod +x install_prerequisites.sh 
RUN ./install_prerequisites.sh
RUN apt-get -y install x11-xserver-utils

#setup ssh connection
RUN echo "" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd

#opencv test app
ENV PROJECT_DIR=/home/openvino/face
ENV MODEL_DIR=$PROJECT_DIR/models
ENV MODEL_NAMES=face-detection-0100,face-detection-0105,face-detection-0106,landmarks-regression-retail-0009
COPY display_test.py $PROJECT_DIR/..

#project folder
RUN mkdir $PROJECT_DIR
RUN mkdir $MODEL_DIR
WORKDIR /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader
RUN for MODEL in $(echo $MODEL_NAMES | tr ',' '\n'); do ./downloader.py --name $MODEL --output_dir $MODEL_DIR --precisions FP32,FP16,INT8; done
# COPY neural-networks /home/openvino/neural-networks

COPY docker_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker_entrypoint.sh
RUN ln -s /usr/local/bin/docker_entrypoint.sh / # backwards compat
ENTRYPOINT ["docker_entrypoint.sh"]

# add apssword to user openvino
RUN echo "openvino:openvino" | chpasswd

# USER openvino
WORKDIR $PROJECT_DIR
CMD ["/bin/bash"]


 # cd /opt/intel/openvino/install_dependencies
 #    9  sudo -E ./opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh
 #   10  source /opt/intel/openvino/bin/setupvars.sh
 #   11  cd /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites
 #   12  sudo ./install_prerequisites.sh
 #   13  cd /opt/intel/openvino/deployment_tools/demo
 #   14  ./demo_squeezenet_download_convert_run.sh
