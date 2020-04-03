# GPQ
Generalized Product Quantization Network for Semi-supervised Image Retrieval - <a href="https://arxiv.org/abs/2002.11281">arxiv</a>  
Accepted to CVPR 2020  
Young Kyun Jang and Nam Ik Cho  

## Get Started

### 1. Install requirements on your environment.
- Ubuntu=16.04
- Tensorflow=1.14
- Tflearn=0.3.2
- Numpy=1.16.6
- Matplotlib=3.2.1

### 2. Preparation.
- Download pretrained model and data in <a href="https://drive.google.com/open?id=1BfyXFvcMMBhD2jWVNF_kFaFE5uNgpqII">here</a>
- Extract them to the `/path/to/GPQ-master`

### 3. Test
- Modify `config.py` to set the path as:  
```
data_dir = '/path/to/GPQ-master/cifar10'
cifar10_label_sim_path = '/path/to/GPQ-master/cifar10/cifar10_Similarity.mat'
model_load_path='/path/to/GPQ-master/models/48bits_example.ckpt'
```
### 4. Train
-
-


# tSNE Visualization

<p align="center"><img src="figures/tSNE.png" width="900"></p>

# Training
