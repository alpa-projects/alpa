# Run Alpa in k8s cloud with InfiniBand (CoreWeave)
To run Alpa in specialized GPU cloud like [CoreWeave](https://coreweave.com/), we will need a few pieces in addition to [default run Alpa in Docker](../README.md):

1. InfiniBand dependencies in Alpa docker image
2. K8s deployment YAML file to declare Ray cluster resources
3. Run NCCL with InfiniBand related environment variables such as `NCCL_IB_HCA`

We will go through each step to show you how to deploy Ray cluster in k8s cloud and run Alpa with InfiniBand.

Note most of the content is re-usable for generic k8s and InfiniBand deployment where CoreWeave is the concrete cloud provider we used as verification.

## Build Alpa docker image

First, build a docker image based on the provided dockerfile:
```bash
docker build -t run-alpa-image -f run_alpa_infiniband.Dockerfile .
```

This docker file added InfiniBand dependencies in addition to the [default run_alpa.Dockerfile](../run_alpa.Dockerfile).

## Tag and push your docker image
Then tag and push your Alpa docker image to a public repository in docker.com.
```bash
docker tag {image_hash} {your_docker}/{image}:{version}
```
```bash
docker push {your_docker}/{image}:{version}
```

## Write cluster.yaml file
Then write your deployment script to use the Alpa docker image you just built in a k8s cloud.
The k8s deployment process can be summarized as the following steps in a nutshell:

1. Define service/headnode/worker roles in the k8s deployment for the Ray cluster.
2. Make physical resource requirements to the k8s cloud regarding GPU/CPU/RAM/InfiniBand/number of replicas.
3. Pull the Alpa docker image you built with Ray.
4. For each container, activate Alpa conda environment and run `ray start` to establish Ray runtime across the cluster.

[Example end to end working YAML file](cluster.yaml)

Change the `TODO` in sample YAML file to match your desired namespace, docker image and resource requirements.

## Deploy to k8s

Then we can use simple idempotent commands to start and terminate your Ray cluster to run Alpa.
```bash
kubectl apply -f cluster.yaml
```

```bash
kubectl delete -f cluster.yaml
```

## Example end-to-end workflow

Once your cluster is started, you should be able to monitor all pods like this:
```
❯ k get pods
NAME                                    READY   STATUS    RESTARTS   AGE
deployment-ray-head-d9dc9cf7f-pkqvz     1/1     Running   0          2m25s
deployment-ray-worker-d66d65c7b-25659   1/1     Running   0          2m24s
deployment-ray-worker-d66d65c7b-6sbpz   1/1     Running   0          2m24s
deployment-ray-worker-d66d65c7b-8smzr   1/1     Running   0          2m24s
```

You can ssh into the headnode for interactive development and job submission.
```bash
kubectl exec --stdin --tty deployment-ray-head-d9dc9cf7f-pkqvz -- /bin/bash -i -l
```

Then activate alpa conda environment:
```bash
conda activate alpa

```

And verify your Ray cluster is running as expected.
```
(alpa) ray@deployment-ray-head-d9dc9cf7f-pkqvz:~$ ray status
======== Autoscaler status: 2022-12-29 10:05:41.200229 ========
Node status
---------------------------------------------------------------
Healthy:
 1 node_a4328576d9fee799a5e6853acba0a6c1e1d8cb7fbabed6a6bab3649a
 1 node_475ed937e3506d7f47ac1abc508e0eb7cde2a270d86a23fad3b9d0b2
 1 node_347bc30b1fe0cc5f5730a6f803018fe2f3b6597226be69580995b436
 1 node_8725d199fd3ef007abb673be6307a233a6f90f1001d8cd29aa873789
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/128.0 CPU
 0.0/32.0 GPU
 0.0/4.0 accelerator_type:A100
 0.00/197.961 GiB memory
 0.00/86.199 GiB object_store_memory
 ```

 ## Environment variables for NCCL

 In order to enable InfiniBand for NCCL communication, you will need a few additional env vars, such as `NCCL_IB_HCA=ibp`. You can see the full list of configurations in [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

 ## Run Alpa's NCCL test

Alpa uses cupy / ray collective / xla to orchestrate NCCL communcation.
You should be able to run the NCCL test [profile_communication](https://github.com/alpa-projects/alpa/blob/5660516ad3a29e5760673e599fc84aa604589a82/benchmark/cupy/profile_communication.py) in

```bash
python profile_communication.py --ib
```

Optionally add `--debug` to show NCCL logs to ensure InfiniBand is indeed used instead of Ethernet, as their AllReduce performance difference is expected to be very significant.

Sample output from a 4 node 8x80GB A100s NVLink cluster:

```
AllReduce: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]	Bytes: 2.00000 GB	Time: 0.04278 s	Bandwidth: 90.59 GB/s
AllReduce: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]	Bytes: 2.00000 GB	Time: 0.03842 s	Bandwidth: 97.59 GB/s
AllReduce: [[0, 3]]	Bytes: 2.00000 GB	Time: 0.01006 s	Bandwidth: 198.82 GB/s
AllReduce: [[0, 4], [1, 5], [2, 6], [3, 7]]	Bytes: 2.00000 GB	Time: 0.00994 s	Bandwidth: 201.30 GB/s
AllReduce: [[0, 2, 4, 6], [1, 3, 5, 7]]	Bytes: 2.00000 GB	Time: 0.01404 s	Bandwidth: 213.71 GB/s
AllReduce: [[0, 1, 2, 3], [4, 5, 6, 7]]	Bytes: 2.00000 GB	Time: 0.01406 s	Bandwidth: 213.31 GB/s
AllReduce: [[0, 1, 2, 3, 4, 5, 6, 7]]	Bytes: 2.00000 GB	Time: 0.01623 s	Bandwidth: 215.60 GB/s

SendRecv: [[0, 1]]	Bytes: 2.00000 GB	Time: 0.00814 s	Bandwidth: 245.59 GB/s
SendRecv: [[0, 31]]	Bytes: 2.00000 GB	Time: 0.15949 s	Bandwidth: 12.54 GB/s
SendRecv: [[0, 1], [2, 3]]	Bytes: 2.00000 GB	Time: 0.00815 s	Bandwidth: 490.84 GB/s
SendRecv: [[0, 28], [1, 29]]	Bytes: 2.00000 GB	Time: 0.17521 s	Bandwidth: 22.83 GB/s
SendRecv: [[0, 30], [1, 31]]	Bytes: 2.00000 GB	Time: 0.17519 s	Bandwidth: 22.83 GB/s
SendRecv: [[0, 28], [1, 29], [2, 30], [3, 31]]	Bytes: 2.00000 GB	Time: 0.17526 s	Bandwidth: 45.65 GB/s
SendRecv: [[0, 24], [1, 25], [2, 26], [3, 27]]	Bytes: 2.00000 GB	Time: 0.17486 s	Bandwidth: 45.75 GB/s
SendRecv: [[0, 24], [1, 25], [2, 26], [3, 27], [4, 28], [5, 29], [6, 30], [7, 31]]	Bytes: 2.00000 GB	Time: 0.17491 s	Bandwidth: 91.48 GB/s
```