#!/bin/bash

#check if docker is running
if [[ "$(systemctl is-active docker)" != "active" ]]; then
  echo "docker service is not running"
  exit 1
fi