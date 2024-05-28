#!/bin/bash
drun() {
    docker run --runtime=nvidia --gpus all tensorplus "$@"
}

drun_no_cuda() {
    docker run tensorplus "$@"
}

helpmsg() {
    echo "Some simple usage instructions:"
    echo "1) drun --help                                ---     Display this message"
    echo "2) drun --no-cuda [arg1] [arg2] ... [argN]    ---     Run the container with the specified arguments and without Nvidia Cuda support"
    echo "3) drun [arg1] [arg2] ... [argN]              ---     Run the container with the specified arguments"
    echo "4) drun --build                               ---     Build the container from the dockerfile"
    echo "5) drun --send [src] [dest]                   ---     Copy a file from the host to the container"
    echo "6) drun --get [src] [dest]                    ---     Copy a file from the container to the host"
}

if [ "$1" == "--help" ]; then
    helpmsg
    exit 0
fi
if [ "$1" == "--no-cuda" ]; then
    shift
    drun_no_cuda "$@"
    exit 0
fi
if [ "$1" == "--build" ]; then
    docker build . -t tensorplus
    exit 0
fi
if [ "$1" == "--send" ]; then
    docker cp "$2" "tensorplus:$3"
    exit 0
fi
if [ "$1" == "--get" ]; then
    docker cp "tensorplus:$2" "$3"
    exit 0
fi
if [ "$1" == "--stay" ]; then
    docker run -dit --runtime=nvidia --gpus all tensorplus
    exit 0
fi
if [ "$1" == "--stay-no-cuda" ]; then
    docker run -dit --name tensorplus tensorplus
    exit 0
fi
if [ "$1" == "--attach" ]; then
    docker attach tensorplus
    exit 0
fi
drun "$@"
