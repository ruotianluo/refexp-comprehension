from __future__ import print_function, division

import sys
import os
import numpy as np
import skimage.io
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe

import util
from captioner import Captioner


vgg_weights_path = './models/VGG_ILSVRC_16_layers.caffemodel'
gpu_id = 0

image_dir = './data/resized_imcrop/'
cached_local_features_dir = './data/referit_local_features/'

image_net_proto = './prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = './prototxt/scrc_word_to_preds_full.prototxt'
vocab_file = './data/vocabulary.txt'

captioner = Captioner(vgg_weights_path, image_net_proto, lstm_net_proto, vocab_file, gpu_id)
batch_size = 100
captioner.set_image_batch_size(batch_size)


metadata_dir = './data/metadata/'
imcrop_dict = util.io.load_json(metadata_dir + 'referit_imcrop_dict.json')
imlist = []
for k, v in imcrop_dict.iteritems():
    imlist = imlist + v

num_im = len(imlist)

# Make dir if not existed
if not os.path.isdir(cached_local_features_dir):
    os.mkdir(cached_local_features_dir)

# Load all images into memory
loaded_images = []
loaded_ids = []
for n_im in range(num_im):

    im = skimage.io.imread(image_dir + imlist[n_im] + '.png')
    # Gray scale to RGB
    if im.ndim == 2:
        im = np.tile(im[..., np.newaxis], (1, 1, 3))
    # RGBA to RGB
    im = im[:, :, :3]
    loaded_images.append(im)
    loaded_ids.append(n_im)

    if (n_im + 1) % 200 == 0 or n_im == num_im - 1:
        print('loading image %d / %d into memory' % (n_im, num_im))
        print('Compute discriptors:')
        # Compute fc7 feature from loaded images, as whole image bbox feature
        descriptors = captioner.compute_descriptors(loaded_images, output_name='fc7')
        for _, idx in enumerate(loaded_ids):
            save_path = cached_local_features_dir + imlist[idx] + '_fc7.npy'
            np.save(save_path, descriptors[_, :])
        loaded_images = []
        loaded_ids = []
        
