# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
This is the artifact for the paper "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning".
We are going to reproduce the main results in the paper.

## Setting up the Environment

Please see the README file provided on hotcrp for the AWS credentials, and write the credentials to the file `~/.aws/credentials`:
``` bash
mkdir -p ~/.aws
echo "[default]" > ~/.aws/credentials
echo "aws_access_key_id = <AWS access key>" >> ~/.aws/credentials
echo "aws_secret_access_key = <AWS secret key>" >> ~/.aws/credentials
```
Please replace the AWS access key and secret key with the provided credentials.

### Install ray on the local environment

Run the following command on your local machine to install ray. We use ray and boto3 to start the cluster.
```
pip install -U "ray[default]" boto3
```
Follow [this page](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support) if you are using an M1 Mac.

### Start the cluster (3 min)
Clone this repository to your local machine and go to `alpa/osdi22_artifact/`. Start the cluster with the following command:

```
ray up -y artifact-cluster.yaml
```

After the cluster is started, you can access the cluster with the following command:
```
ray attach artifact-cluster.yaml
```

To terminate the cluster, run the following command:
```
ray down -y artifact-cluster.yaml
```

### Check cluster status (5 min)
1. Connect to the cluster with
  ```
  ray attach artifact-cluster.yaml
  ```
  and move to the following directory:
  ```
  cd efs/alpa/osdi22_artifact/
  ```
2. Run
  ```
  python3 -c "import ray; ray.init(address='auto'); print('#GPU:', ray.cluster_resources()['GPU'])"
  ```

  You should be able to see it outputs
  ```
  ...
  #GPU: 32.0
  ```

  If the number of GPUs is less than 32, wait couple minutes and check again for the cluster to be ready.
3. Run
  ```
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

### GPT (30 min)
```
python3 gen_data_e2e.py --model gpt
python3 plot_e2e.py --model gpt
```
This outputs Figure 8 (a).

### MoE (30 min)
```
python3 gen_data_e2e.py --model moe
python3 plot_e2e.py --model moe
```
This outputs Figure 8 (b).

### Wide-ResNet (30 min)
```
python3 gen_data_e2e.py --model wresnet
python3 plot_e2e.py --model wresnet
```
This outputs Figure 8 (c).


Since Megatron-LM and DeepSpeed requires different setups, we provide instructions to generate their performance numbers 
separately as below.
### Megatron-LM Performance on GPT (30 mins)
Following the instructions in [megatron/setup.md](setup/setup.md) to run Megatron-LM benchmarking.
This outputs the red bar corresponding to `Megatron-LM` in Figure 8 (a).

### DeepSpeed Performance on MoE (30 mins)
Following the instructions in [deepspeed/setup.md](setup/setup.md) to run DeepSpeed benchmarking.
This outputs the brown bar corresponding to `DeepSpeed` in Figure 8 (b).

## Intra-op Ablation Study (Figure. 9)

### GPT  (30 min)
```
python3 gen_data_intra_ab.py --model gpt
python3 plot_intra_ab.py --model gpt
```
This outputs Figure 9 (a).

### MoE (45 min)
```
python3 gen_data_intra_ab.py --model moe
python3 plot_intra_ab.py --model moe
```
This outputs Figure 9 (b).

### Wide-ResNet (30 min)
```
python3 gen_data_intra_ab.py --model wresnet
python3 plot_intra_ab.py --model wresnet
```
This outputs Figure 9 (c).

## Inter-op Ablation Study (Figure. 10)

```
python3 gen_data_inter_ab.py
python3 plot_inter_ab.py
```

After the experiments, please terminate the cluster with the following command on your local machine to reduce the cost of AWS charges:
```
ray down -y artifact-cluster.yaml
```

## (Optional) Running the Search
