"""
Copy from Natural ....

Modify only one part, cache all the proposals instead of only proposals for test images
"""

from __future__ import division, print_function
import argparse

import sys
import numpy as np
import skimage.io
sys.path.append('./external/caffe-natural-language-object-retrieval/python/')
sys.path.append('./external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os

import util
from captioner import Captioner
import retriever

parser = argparse.ArgumentParser()
# Data input settings
parser.add_argument('--start_from', type=int, default=0,
                help='start from which image')
args = parser.parse_args()

import time
start = time.time()


bbox_dir = './data/referit_visualize/annotated/'

gpu_id = 0  # the GPU to test the SCRC model

################################################################################

image_dir = './datasets/ReferIt/ImageCLEF/images/'
proposal_dir = './data/referit_edgeboxes_top100/'

vocab_file = './data/vocabulary.txt'

# utilize the captioner module from LRCN
image_net_proto = './prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = './prototxt/scrc_word_to_preds_full.prototxt'
vgg_weights_path = './models/VGG_ILSVRC_16_layers.caffemodel'
captioner = Captioner(vgg_weights_path, image_net_proto, lstm_net_proto, vocab_file, gpu_id)
captioner.set_image_batch_size(50)
#vocab_dict = retriever.build_vocab_dict_from_captioner(captioner)

# Load image and caption list
image_dir = './datasets/ReferIt/ImageCLEF/images/'

imlist = util.io.load_str_list('./data/split/referit_all_imlist.txt')
num_im = len(imlist)

# Load candidate regions (bounding boxes)
candidate_boxes_dict = {imname: None for imname in imlist}
for n_im in range(num_im):
    if n_im % 1000 == 0:
        print('loading candidate regions %d / %d' % (n_im, num_im))
    imname = imlist[n_im]
    # from edgebox
    proposal_file_name = imname + '.txt'
    boxes = np.loadtxt(proposal_dir + proposal_file_name)
    boxes = boxes.astype(int).reshape((-1, 4))

    candidate_boxes_dict[imname] = boxes

sample_im = num_im

cached_proposal_features_dir = './data/referit_proposal_features/'
if not os.path.isdir(cached_proposal_features_dir):
    os.mkdir(cached_proposal_features_dir)


for n_im in range(args.start_from, sample_im):
    if time.time() - start > 3600 *3.5: #3.5 hours
        f = open('tmp.sh', 'w')
        f.write('#! /bin/sh\n')
        f.write('python ./exp-referit/cache_edgebox_feature.py --start_from ' + str(n_im))
        f.flush()
        f.close()
        os.system("python sbatch.py -T 1 -J cache -sh tmp.sh -C titanx")
        break

    print('Cache edgebox image %d / %d' % (n_im, num_im))
    imname = imlist[n_im]

    candidate_boxes = candidate_boxes_dict[imname]

    if candidate_boxes.shape[0] == 0:
        continue

    im = skimage.io.imread(image_dir + imname + '.jpg')
    imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]

    # Compute local descriptors (local image feature + spatial feature)
    descriptors = retriever.compute_descriptors_edgebox(captioner, im,
                                                        candidate_boxes,'fc7') # (100,4096)
    spatial_feats = retriever.compute_spatial_feat(candidate_boxes, imsize) # (100,8)
    np.savez(cached_proposal_features_dir+imname, spatial_feat=spatial_feats,local_feature=descriptors)
