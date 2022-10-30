#!/bin/bash --login

#conda init bash

conda activate alpa 

echo y|ray start --head 
cd benchmark/alpa
