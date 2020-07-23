#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit ; pwd -P )"

#make output file
out_name='exp_2_data.txt'
if [[ -f "$out_name" ]]; then
    rm "$out_name"
fi
touch "$out_name"

face_detection="./models/intel/face-detection-0100/FP32/face-detection-0100.xml"
landmarks_detection="./models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
reidentification_model="./models/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"

# all supported devices
devices='CPU MYRIAD GPU'
IP=$(hostname -I | awk -F ' ' '{print $1}' | awk -F '.' '{print $4}')
if [[ "$IP" == 206 || "$IP" == 207 ]]; then
  devices='MYRIAD'
  face_detection="./model_library/intel/face-detection-0105/FP32/face-detection-0105.xml"
  landmarks_detection="./model_library/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml"
  reidentification_model="./model_library/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml"
fi

# all supported people count
people_counts='1 2 4 8'
# all supported resolutions
resolutions='240 360 480 720 1080 1440 2160'

# loop through tests
for dev in $devices; do
  for count in $people_counts; do
    for res in $resolutions; do
      echo "* ${dev}, ${count}, ${res}"
      if [[ "$IP" == 206 || "$IP" == 207 ]]; then
        cd ..
      fi
      times=$( (python3 face_recognition.py -d "$dev" -dm "$face_detection" -dm_t 0.65 -lm "$landmarks_detection" \
      -rm "$reidentification_model" -on -t --input_video  "./test_videos/face_${count}/face_${count}_${res}p.mp4") 2> /dev/null)
      echo "$times"
      if [[ "$IP" == 206 || "$IP" == 207 ]]; then
        cd "$SCRIPTPATH" || exit
      fi
      for time in $(echo "$times" | sed "s/;/ /g"); do
        echo "$dev;$count;$res;$time" >> "$out_name"
      done
    done
  done
done


echo "experiment 2 done"