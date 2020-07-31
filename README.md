HALVÝ SÚBOR APLIKÁCIE MÁ NÁZOV: face_ecognition.py

!!!JE POTREBA UPRAVIŤ CESTY PODĽA ZARIADENIA N AKTOROM TO JE SPUSTANÉ!!!!
neplatí pri veciach vovnútri kontajneru

aplikácia spustitelná len s správne nainštalovaým a prostredím OPENVINO
Priečinok Docker obsahuje potrebensubory na vytvorenie spravneho OPENVINO prstredia

Docker/docker_builder:
-no arguments means build all for PC platform
-o_base = only bare bone container as provided by Intel
-o_dev = builds ontop of base sets up whole env.
-nvidia = buildas container for TF Lite to execute test 1 on GPU Nvidia
-r1 and -r2 = experimental (not used in solution)
Docker/docker_runner:
runns docker containers, supports 2 args, second is optional 
-openvino, -raspberry, -nvidia without -it starts container and lets it run on background,
                                with -it starts interactive session too 
na novom systeme spustit:
./Docker/docker_builder.sh
./Docker/docker_runner.sh -openvino
if this project is placed outside container run before launch flowing script:
./support_scripts/copy_project_to_remote.sh -a 

on properly set environment (I recommend docker container) launch by:
python3 face_ecognition.py
-d
MYRIAD
-dm
/home/openvino/face/models/intel/face-detection-0100/FP32/face-detection-0100.xml
-dm_t
0.65
-lm
/home/openvino/face/models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml
-rm
/home/openvino/face/models/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml
--input_camera
0
-od
-t