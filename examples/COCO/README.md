## Unet on Coco segmentation task

Trains Unet on the COCO dataset.

Adopted from https://github.com/google-research/scenic for Unet and 
https://github.com/baudcode/tf-semantic-segmentation for COCO dataset, 
we add the following modifications to better demonstrate the training speedup by Alpa. And we could also show Alpa's scalibility here.
- We allow the number of convolution layers at each height to be changed which are 2 for default unet configuration, making it more flexible to scale up. 
- Use a large batch size.
- ...

### Requirements
* Tensorflow 2.8.0
* imageio 2.16.0
* opencv-python 4.5.5.62

### Scalability 

| # of GPU   | 2 | 4 | 8 | 16 | 32 |
| :------ | -----: | -------: | -------------: | ----------: | :---------------------------------------- |
| config(bsz, channelsz, blockcnt): |  (256, (64, 64, 96, 128), (6, 6, 6, 6, 6)) | (256, (64, 128, 256, 256), (10, 10, 10, 10, 10)) | (64, (384, 384), (32, 64, 128, 256), (8, 8, 8, 8, 8)) | | |
| compilation time(s): | 155.21 | 6249.41 | 805.98 | | |
| one batch execution time(s): | 16.63 | 19.27 | 1.96 | | |
| tflops: | 5.55 | 9.58 | 3.23 | | |
| param_count(B): | 0.007 | 0.052 | 0.032 | | |
| max_mem_allocated(G): | 33.89 | 15.59 | 1.88| | |
### How to run

- To run the naive example for correctness test:

> `python train_unet_coco.py --num-gpu 2`

- To reproduce experiments results, run command below. A slurm script is also provided in ./scripts. To modify model and other configurations, please refer to unet_suite.py. 

> `python3 -u unet_benchmark_3d.py --suite unet.test --exp_name auto_2_gpus --num-hosts 1 --num-devices-per-host 2`



