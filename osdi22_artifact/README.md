# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
This is the artifact for the paper "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning".
We are going to reproduce the main results in the paper.

## Setting up the Environment
Please use the provided instructions to log in to the AWS cluster set up by authors.

## End-to-end Performance (Figure. 8)

### GPT
```
python3 gen_data_e2e.py --model gpt
python3 plot_e2e.py --model gpt
```

### MoE
```
python3 gen_data_e2e.py --model moe
python3 plot_e2e.py --model moe
```

### Wide-ResNet
```
python3 gen_data_e2e.py --model wresnet
python3 plot_e2e.py --model wresnet
```

## Intra-op Ablation Study (Figure. 9)

### GPT
```
python3 gen_data_intra_ab.py --model gpt
python3 plot_intra_ab.py --model gpt
```

### MoE
```
python3 gen_data_intra_ab.py --model moe
python3 plot_intra_ab.py --model moe
```

### Wide-ResNet
```
python3 gen_data_intra_ab.py --model wresnet
python3 plot_intra_ab.py --model wresnet
```

## Inter-op Ablation Study (Figure. 10)

```
python3 gen_data_inter_ab.py
python3 plot_inter_ab.py
```
