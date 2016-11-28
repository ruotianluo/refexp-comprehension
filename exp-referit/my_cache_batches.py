from __future__ import print_function, division

import os
import numpy as np

import util
import cPickle as pickle
import retriever

parser = argparse.ArgumentParser()
# Data input settings
parser.add_argument('--dataset', type=str, default='train',
                help='train,val,test,trainval')
args = parser.parse_args()

dataset = args.dataset

trn_imlist_file = './data/split/referit_'+dataset+'_imlist.txt'

image_dir = './datasets/ReferIt/ImageCLEF/images/'
resized_imcrop_dir = './data/resized_imcrop/'
cached_context_features_dir = './data/referit_context_features/'
cached_local_features_dir = './data/referit_local_features/'
cached_proposal_features_dir = './data/referit_proposal_features/'

imcrop_dict_file = './data/metadata/referit_imcrop_dict.json'
imcrop_bbox_dict_file = './data/metadata/referit_imcrop_bbox_dict.json'
imsize_dict_file = './data/metadata/referit_imsize_dict.json'
query_file = './data/metadata/referit_query_dict.json'
vocab_file = './data/vocabulary.txt'

N_batch = 50  # batch size during training
T = 20  # unroll timestep of LSTM

imset = set(util.io.load_str_list(trn_imlist_file))
vocab_dict = retriever.build_vocab_dict_from_file(vocab_file)
query_dict = util.io.load_json(query_file)
imsize_dict = util.io.load_json(imsize_dict_file)
imcrop_bbox_dict = util.io.load_json(imcrop_bbox_dict_file)

train_pairs = []
for imcrop_name, des in query_dict.iteritems():
    imname = imcrop_name.split('_', 1)[0]
    if imname not in imset:
        continue
    if not os.path.isfile(os.path.join(cached_proposal_features_dir,imname + '.npz')):
        continue
    imsize = np.array(imsize_dict[imname])
    bbox = np.array(imcrop_bbox_dict[imcrop_name])
    bbox_feat = retriever.compute_spatial_feat(bbox, imsize)
    train_pairs += [(imcrop_name, retriever.sentence2vocab_indices(d, vocab_dict), 
        bbox_feat, imname) for d in des]

pickle.dump(train_pairs, open('data/'+dataset+'_pairs.pkl', 'w'))
