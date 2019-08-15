# LCA-CNN

Code for paper "Learning Cascade Attention for Fine-grained Image Classification", which currently 
under review at Elsevier Journal of Neural Networks(NN).


## File description
1. bootstrap.py  
Train/test split for CUB-200-2011 dataset.

2. image_rotate.py  
Rotate training set for data augmentation.

3. model.py  
Core model code.

4. inception_train.py  
Training and validation code.

5. inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5  
The iNaturalist pre-train parameters (converted from [Link](https://github.com/richardaecn/cvpr18-inaturalist-transfer)).

6. weights1/weights-gatp-two-stream-inception_v3-006-0.9080.hdf5  
Trained model for CUB-200-2011 dataset. (for reproduction).  
To unzip file:
```
zip -s- weights1.zip -O weights111.zip
unzip weights111.zip
```

## Dependencies:
+ Python (3.6)
+ Keras (2.1.5)
+ Tensorflow (1.10.0)


------






