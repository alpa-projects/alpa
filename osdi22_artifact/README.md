# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
This is the artifact for the paper "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning".
We are going to reproduce the main results in the paper.

## ❗️❗️Update❗️❗️

1. The original AWS access keys provided in hotcrp README is wrong. Please find the updated access keys in the comments section on hotcrp.
2. We experienced some concurrency issues that many of you are trying to ray up or ray down at the same time, and connecting to the same cluster. Please modify [this line](artifact-cluster.yaml#L2) in [artifact-cluster.yaml](artifact-cluster.yaml) to a unique name as below. This is to make sure that each of you can get a unique cluster.
    ```yaml
    # Before modification
    # A unique identifier for the head node and workers of this cluster.
    cluster_name: artifact-cluster

    # After modification example
    # A unique identifier for the head node and workers of this cluster.
    cluster_name: artifact-cluster-reviewer-A
    ```
   Note that even with this change, the underlying EFS (i.e. NFS on AWS) where we put our code and data is still shared among all the machines. We believe the chance you meet concurrency issue on EFS is small, but please let us know if you meet any.

## Setting up the Environment
All experiments will be run on an AWS cluster. We provide credentials for you to launch and connect to the cluster.

### Install AWS Credentials (3 mins)
Replace `<access key id>` and `<secrete access key>` in the following commands with the values provided on hotcrp.  
Backup your old credentials before running the following commands if necessary.
```bash
mkdir -p ~/.aws
echo "[default]" > ~/.aws/credentials
echo "aws_access_key_id = <access key id>" >> ~/.aws/credentials
echo "aws_secret_access_key = <secret access key>" >> ~/.aws/credentials
```

### Install Ray (5 mins)
We use ray and boto3 to launch the cluster.
Run the following command on your local machine to install ray. 
```bash
pip3 install -U "ray[default]" boto3
```
Follow [this page](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support) if you are using an M1 Mac.

### Launch the Cluster (5 mins)
Clone this repository to your local machine and go to `alpa/osdi22_artifact/`. Start the cluster by
```bash
ray up -y artifact-cluster.yaml
```

After the cluster is started, you can log in to the head node of the cluster by
```bash
ray attach artifact-cluster.yaml
```

You can use the following command to shutdown the cluster on your local machine. DO NOT RUN IT NOW.
**Make sure to run it after your finish the artfact evaluation or want to pause.**
The price of this AWS cluster is expensive: $100/hour paid by us.
```bash
ray down -y artifact-cluster.yaml
```

### Check the Cluster Status (5 mins)
1. Connect to the cluster by
  ```bash
  ray attach artifact-cluster.yaml
  ```
  and go to the following directory:
  ```bash
  cd efs/alpa/osdi22_artifact/
  ```
2. Wait until all worker nodes are launched. This can take a couple of minutes.
  ```bash
  python3 wait_workers.py
  ```
3. Test installation by
  ```bash
  python3 test_install.py
  ```

  You should be able to see it outputs
  ```
  ...
  Ran 2 tests in 177.621s

  OK
  ```

## End-to-end Performance (Figure. 8)
The profiling and optimization for all data points in our paper take more than 10 hours to run.
To make the artifact evaluation feasible in a reasonable amount of time, we skip some profiling and
optimization procedures, but directly load the solutions found by our system and measure their throughputs.
Optionally, if you want to run the profiling and optimization from scratch by yourself, follow the
instructions in the section "Running the Search" below.

### GPT (30 mins)
```bash
python3 gen_data_e2e.py --model gpt
python3 plot_e2e.py --model gpt
```
This outputs `e2e_gpt.pdf`, or Figure 8 (a) in the paper.

You can run `ray rsync-down artifact-cluster.yaml "~/efs/alpa/osdi22_artifact/*.pdf" .` **on your local machine**
to copy the figures to your local machine for review.

### MoE (30 mins)
```bash
python3 gen_data_e2e.py --model moe
python3 plot_e2e.py --model moe
```
This outputs `e2e_moe.pdf`, or Figure 8 (b) in the paper.

### Wide-ResNet (60 mins)
```bash
python3 gen_data_e2e.py --model wresnet
python3 plot_e2e.py --model wresnet
```
This outputs `e2e_wresnet.pdf`, or Figure 8 (c) in the paper.

## Intra-op Ablation Study (Figure. 9)

### GPT  (30 mins)
```bash
python3 gen_data_intra_ab.py --model gpt
python3 plot_intra_ab.py --model gpt
```
This outputs `intra_ab_gpt.pdf`, or Figure 9 (a) in the paper.

### MoE (45 mins)
```bash
python3 gen_data_intra_ab.py --model moe
python3 plot_intra_ab.py --model moe
```
This outputs `intra_ab_moe.pdf`, or Figure 9 (b) in the paper.

### Wide-ResNet (30 mins)
```bash
python3 gen_data_intra_ab.py --model wresnet
python3 plot_intra_ab.py --model wresnet
```
This outputs `intra_ab_wresnet.pdf`, or Figure 9 (c) in the paper.

## Inter-op Ablation Study (Figure. 10)

### GPT and Wide-ResNet (60 mins)
```bash
python3 gen_data_inter_ab.py --model gpt
python3 gen_data_inter_ab.py --model wresnet
python3 plot_inter_ab.py
```
This outputs Figure 10 in the paper.

## (Optional) Running the Search
To run the full search procedure, use the commands below.

### Run One Case of GPT (30 mins)
Run one case for 16 GPUs.
```
python3 gen_data_e2e.py --model gpt --search --cluster--size 16
```

### Run All Cases of GPT (2 hours)
Run all cases for 1-32 GPUs.
```
python3 gen_data_e2e.py --model gpt --search
```
You can also use similar commands for other models.

### About the Profiling Database
Note that we provide a profiling database as the performance model for the database in the AMI generated by us (`prof_database_osdi22_artifact.pkl`) since the process of generating the profiling database can take several hours. If you would like to generate the database, use the command below:
```bash
cd ~/efs/alpa/benchmark/alpa
python3 gen_prof_database.py --comm-size-max 30 --filename your_prof_database.pkl
```

## (Optional) Running the Megatron-LM and Deepspeed
The performance numbers of Megatron-LM and Deepspeed in the previous plots are
loaded from our previous measurments in `results_e2e_ref.tsv`.
The data is obtained as follows.

### Megatron-LM Performance on GPT (30 mins)
Follow the instructions in [megatron/README.md](megatron/README.md) to run Megatron-LM benchmarking.
This generates data for the red bar corresponding to `Megatron-LM` in Figure 8 (a).

### DeepSpeed Performance on MoE (30 mins)
Follow the instructions in [deepspeed/README.md](deepspeed/README.md) to run DeepSpeed benchmarking.
This generates data for the brown bar corresponding to `DeepSpeed` in Figure 8 (b).
