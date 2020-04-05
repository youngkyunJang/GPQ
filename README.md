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
- Download pretrained model and data from <a href="https://drive.google.com/open?id=1BfyXFvcMMBhD2jWVNF_kFaFE5uNgpqII">here</a>.
- Extract them to the `/path/to/GPQ-master`.
- Make sure the tree seems as:  
```
|--GPQ-master
   |--cifar10
      |--batches.meta
      |--data_batch_1
      |--data_batch_2
      |--data_batch_3
      |--data_batch_4
      |--data_batch_5
      |--test_batch
      |--cifar10_Similarity.mat
   |--models
      |--48bits_example.ckpt.data-00000-of-00001
      |--48bits_example.ckpt.index
      |--48bits_example.ckpt.meta
      |--ImageNet_pretrained.mat
```
### 3. Test
 - From cifar10 dataset, we use 1,000 images for query and 54,000 images to build retrieval database.
 - We provide 48bits pretrained model as an example.
 - Make sure to set proper path in `config.py`.
 - run `Demo.py`, it will show the retrieval result with mAP and stores visualized search results of a randomly extracted query.
 - Example
 
 
 
### 4. Train


# tSNE Visualization

<p align="center"><img src="figures/tSNE.png" width="900"></p>

# Training
