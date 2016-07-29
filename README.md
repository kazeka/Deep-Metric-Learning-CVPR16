# Deep Metric Learning via Lifted Structured Feature Embedding
This repository has the source code and the Stanford Online Products dataset for the paper "Deep Metric Learning via Lifted Structured Feature Embedding" (CVPR16). The paper preprint is available on [arXiv](http://arxiv.org/abs/1511.06452). If you just need the Caffe code, check out the [Submodule](https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16). For the loss layer implementation, look at [here](https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp).

## Citing this work
If you find this work useful in your research, please consider citing:

    @inproceedings{songCVPR16,
        Author = {Hyun Oh Song and Yu Xiang and Stefanie Jegelka and Silvio Savarese},
        Title = {Deep Metric Learning via Lifted Structured Feature Embedding},
        Booktitle = {Computer Vision and Pattern Recognition (CVPR)},
        Year = {2016}
    }

## Installation
1. Install prerequsites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))
2. Compile the `Caffe-Deep-Metric-Learning-CVPR16` Github submodule.

## Training procedure
1. Download pretrained GoogLeNet model from [here](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
2. Download the ILSVRC12 ImageNet mean file for mean subtraction. Refer to Caffe the ImageNet examples  [here](https://github.com/BVLC/caffe/tree/master/examples/imagenet).
3. Generate the LMDB file to convert the training set of images to the DB format. Example scripts are in `code/` directory.
 * Modify and run `code/compile.m` to mex compile the cpp files used for LMDB generation.
 * Modify `code/config.m` to set save paths.
 * Run `code/gen_caffe_dataset_multilabel_m128.m` to start the LMDB generation process.
4. Create the `model/train*.prototxt` and `model/solver*.prototxt` files. Please refer to the included `*.prototxt` files in `model/` directory for examples. You also need to provide the path to the ImageNet mean file (usually called `imagenet_mean.binaryproto`) you downloaded in step 2.
5. Inside the caffe submodule, launch the Caffe training procedure.
`caffe/build/tools/caffe train -solver [path-to-training-prototxt-file] -weights [path-to-pretrained-googlenet] -gpu [gpuid]`

## Feature extraction after training
1. Modify and run `code/gen_caffe_validation_imageset.m` to convert the test images to LMDB format.
1. Modify the test set path in `model/extract_googlenet*.prototxt`.
2. Modify the model and test set path and run `code/compute_googlenet_distance_matrix_cuda_embeddings_liftedstructsim_softmax_pair_m128.py`.

## Clustering and Retrieval evaluation code
1. Use `code/evaluation/evaluate_clustering.m` to evaluate the clustering performance.
2. Use `code/evaluation/evaluate_recall.m` to evaluate recall@K for image retrieval.

## Stanford Online Products dataset
You can download the Stanford Online Products dataset (2.9G) from ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip
* We also have the text meta data for each product images. Please let us know if you're interested in using them.

## Our Pre-trained Models
You can download our pre-trained models on the Cars196 dataset, the CUB200 dataset and the Online Products dataset (265M) from ftp://cs.stanford.edu/cs/cvgl/pretrained_models.zip

## Octave build instructions
To be able to use Octave instead of Matlab to do data preparation, Octave needs to be built with 64 bit index support. This means that the underlying numerics library (such as OpenBLAS or Atlas) needs to support that as well. One combination that works is OpenBLAS + Octave.
1. Build OpenBLAS with the following options: `BINARY64=1 INTERFACE64=1 make`
2. Then Octave can be configured as: `./configure --enable-64 --with-blas=<path to libopenblas.a>`

## Licence
MIT Licence
