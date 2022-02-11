#!/bin/bash

START_IMAGE="nvcr.io/nvidia/pytorch:21.10-py3"
BUILD_NAME="localdev/darknetonnx:21.10-py3"
USERNAME="dev"
ROOT_DIR="/home/$USERNAME"
#ROOT_DIR="/root"

if [ $1 ]
then
    DOCKER_NAME=$1
    START_CMD="/bin/bash"
else
    DOCKER_NAME="python_dev"
    START_CMD="/bin/bash"
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
pushd $SCRIPT_DIR
docker build -f Dockerfile -t $BUILD_NAME \
            --build-arg=BASE_IMAGE=$START_IMAGE \
            --build-arg=USER_NAME=$USERNAME \
            .
popd

