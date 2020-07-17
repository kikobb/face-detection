FROM openvino:dev_base

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
# add apssword to user openvino
RUN echo "openvino:openvino" | chpasswd

#########################
# INITIALIZE ENVIROMENT #
# (compile lib, set paths, ...)
WORKDIR /opt/intel/openvino/install_dependencies/
RUN chmod +x install_openvino_dependencies.sh
RUN ./install_openvino_dependencies.sh
WORKDIR /opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/
RUN chmod +x install_prerequisites.sh 
RUN ./install_prerequisites.sh
RUN apt-get -y install x11-xserver-utils

################################
# CREATE STRUCTURE, COPY FILES #
ENV PROJECT_DIR=/home/openvino/face
ENV MODEL_DIR=$PROJECT_DIR/models
ENV MODEL_NAMES=face-detection-0100,face-detection-0105,landmarks-regression-retail-0009,face-reidentification-retail-0095 
#,face-detection-0106
ENV MOBILENET_V2_DIR=mobilenet_v2

# APPLICATIONS

#opencv test display app
COPY display_test.py $PROJECT_DIR/..

#demo app
RUN mkdir $PROJECT_DIR

#experimetn 1
RUN mkdir $PROJECT_DIR/exp_1

# MODELS
# download intel models
RUN mkdir $MODEL_DIR
WORKDIR /opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader
RUN for MODEL in $(echo $MODEL_NAMES | tr ',' '\n'); do ./downloader.py --name $MODEL --output_dir $MODEL_DIR --precisions FP32,FP16,INT8; done
# COPY neural-networks /home/openvino/neural-networks

# import mobilenet_v2 checkpoints and convert to IR
WORKDIR $MODEL_DIR/tmp_mobilenet
RUN mkdir ../$MOBILENET_V2_DIR
COPY /$MOBILENET_V2_DIR/mobilenet*.tgz $MODEL_DIR/tmp_mobilenet/
# unpac only .pb file (${file%.*} - pattern expansion)
RUN for file in $(ls | grep .tgz); do \
mkdir ../$MOBILENET_V2_DIR/${file%.*}; \
tar -xzf $file -C ../$MOBILENET_V2_DIR/${file%.*} --wildcards '*.pb'; \
done; 
WORKDIR $MODEL_DIR/$MOBILENET_V2_DIR
RUN rm -rf ../tmp_mobilenet
#convert to IR via OpenVino model optimizer
ENV OPTIMIZER=/opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer/mo_tf.py
RUN for FOLDER in $(ls); do \
python3 $OPTIMIZER --input_model $FOLDER/$FOLDER\_frozen.pb --input_shape [1,$(echo $FOLDER | cut -d '_' -f4),$(echo $FOLDER | cut -d '_' -f4),3] --output_dir $FOLDER --silent; \
done;

###############
# ENTRY POINT #
#sert up entrypoint script
COPY support_scripts/docker_entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker_entrypoint.sh
# backwards compat
RUN ln -s /usr/local/bin/docker_entrypoint.sh / 
ENTRYPOINT ["docker_entrypoint.sh"]


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
