#!/bin/bash


if [[ "$1" == "--help" || "$1" == "help" || "$1" == "h" ]]; then
  echo "HELP:"
  echo "first argument: path to .env file"
  exit
fi

if [[ "$#" != 1 ]]; then
  echo "No first argument: path to .env file specified"
  exit 1
fi

#source /opt/intel/openvino/bin/setupvars.sh
PROJECT_PATH="$(grep ^PROJECT_PATH= "$1" | awk -F = '{print $2}' | sed -e "s/'//g")"
echo "before"
printenv | grep PROJECT_PATH
export PROJECT_PATH
echo "after"
printenv | grep PROJECT_PATH
