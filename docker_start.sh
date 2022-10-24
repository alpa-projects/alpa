#!/bin/bash --login

a=$1

sudo docker run --gpus ${a} -v $(pwd):/build --rm --shm-size=10.24gb -it tntnn/alpa:0.1