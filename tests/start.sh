#!/usr/bin/env bash

# Profile DHNE with nsight-systems or nsight-compute

function run_help() { #HELP Display this message:\nZHEN help
    sed -n "s/^.*#HELP\\s//p;" < "$1" | sed "s/\\\\n/\n\t/g;s/$/\n/;s!ZHEN!${1/!/\\!}!g"
    exit 0
}

NSYS_SESS_NAME=profile-zhen

function run_nsys() { #HELP Profile ZHEN with nsight-systems:\nZHEN nsys
    # /usr/local/cuda/nsight-systems-2021.3.2/bin/nsys
    # NSYS=`command -v nsys`
    # NSYS="/usr/local/cuda-11.6/nsight-systems-2021.5.2/bin/nsys"
    PREVIOUS=`sudo cat /proc/sys/kernel/perf_event_paranoid`
    echo 2 | sudo tee /proc/sys/kernel/perf_event_paranoid
    NSYS="/opt/nvidia/nsight-systems/2022.3.4/target-linux-x64/nsys"

    sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH -E \
        $NSYS profile \
        --trace cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --sample system-wide \
        --nic-metrics=true \
        --run-as $USER \
        --wait all \
        --session-new="${NSYS_SESS_NAME}" \
        "$@"

    echo $PREVIOUS | sudo tee /proc/sys/kernel/perf_event_paranoid
    # su $USER -c "$*"
    # --export=json \
    # --ftrace \
    # --ftrace-keep-user-config \
    # --process-scope system-wide \
}

function run_stop_nsys() {
    NSYS="/opt/nvidia/nsight-systems/2022.3.4/target-linux-x64/nsys"

    sudo $NSYS stop --session="${NSYS_SESS_NAME}"
}

function run_ncu() { #HELP Profile ZHEN with nsight-compute:\nZHEN ncu
    # /usr/local/cuda/nsight-compute-2021.2.2/ncu
    NCU=`command -v ncu`
    CUDA_INJECTION64_PATH=none \
    LD_LIBRARY_PATH=/usr/local/cuda/nsight-compute-2021.2.2/target/linux-desktop-glibc_2_11_3-x64:$LD_LIBRARY_PATH \
    $NCU \
        --target-processes all \
        --set full \
        --nvtx \
        -o report -f --import-source yes \
        "$@"
}

function run_local() { #HELP Run ZHEN in local mode:\nZHEN local
    "$@"
}

[[ -z "${1-}" ]] && run_help "$0"
case $1 in
    local|nsys|stop_nsys|ncu) CMD=run_"$@" ;;
    *) run_help "$0" ;;
esac

$CMD

# /home/cjr/miniconda3/envs/alpa/bin/ray start --head --dashboard-host 0.0.0.0 --num-gpus 8