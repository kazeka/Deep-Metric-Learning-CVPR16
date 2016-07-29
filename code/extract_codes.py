import caffe
import cPickle as pickle
import numpy as np

caffe.set_mode_gpu()

net = caffe.Net('model/train_val_googlenet_finetune_liftedstructsim_softmax_pair_m128_multilabel_embed128.prototxt',
                'model/bloom/snapshot_portfolio_embed128_baselr_1E4_iter_40000.caffemodel',
                caffe.TEST)


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load('caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1, 3, 227, 227)

val = [i.split()[0] for i in open('/home/rnd/src/bloom/data-37/test.txt', 'r').read().splitlines()]

embeddings = []

for v in val:
  image = caffe.io.load_image(v)
  net.blobs['data'].data[...] = transformer.preprocess('data', image)
  net.forward()
  embeddings.append(net.blobs['fc_embedding'].data[0].flatten())

with open('bloom_embeddings_40K.pkl', 'w') as fp:
  pickle.dump(embeddings, fp)
