# Referring expression comprehension on ReferIt(RefClef)
This repository contains a similar model as what is called supervised model in following paper:

* Rohrbach, Anna, et al. "Grounding of textual phrases in images by reconstruction." arXiv preprint arXiv:1511.03745 (2015).*

The preprocessing part is borrowed from [link](https://github.com/ronghanghu/natural-language-object-retrieval) and [link](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow)

THe difference to the original paper is: I use logistic loss instead of softmax; and I use bi-directional LSTM; adn I didn't use batch normalization.

The best result I can get on test is around 31, which is the state-of-art.

## Installation

I uses torch to train, and caffe to do preprocessing(copied from [link](https://github.com/ronghanghu/natural-language-object-retrieval) and [link](https://github.com/andrewliao11/Natural-Language-Object-Retrieval-tensorflow))

Torch packages requirement: dpnn. rnn, cbp, torchx, nngraph

Caffe installation:

1. Run `./external/download_caffe.sh` to download the SCRC Caffe version for this experiment. It will be downloaded and unzipped into `external/caffe-natural-language-object-retrieval`. This version is modified from the [Caffe LRCN implementation](http://jeffdonahue.com/lrcn/).
2. Build the SCRC Caffe version in `external/caffe-natural-language-object-retrieval`, following the [Caffe installation instruction](http://caffe.berkeleyvision.org/installation.html). **Remember to also build pycaffe.**

## Train and evaluate model on ReferIt Dataset
1. Download the ReferIt dataset: `./datasets/download_referit_dataset.sh`.
2. Download pre-extracted EdgeBox proposals: `./data/download_edgebox_proposals.sh`.
3. Preprocess the ReferIt dataset to generate metadata needed for training and evaluation: `python ./exp-referit/preprocess_dataset.py`.
4. Cache the scene-level contextual features to disk: `python ./exp-referit/cache_referit_context_features.py`.
5. Cache the local-level features for ground truth regions to disk: `python ./exp-referit/cache_referit_local_features.py`.
6. Cache the local-level features for edgebox regions to disk: `python ./exp-referit/cache_edgebox_feature.py`.
7. python my_cache_batches.py --dataset train/val/test to get train/val/test pairs

To train the model, run

`th train.lua -visnet_type rt -loss_type logistic -learning_rate 1e-4 -save_checkpoint_every 40000 -val_images_use 15000 -load_best_score 0`

To eval the model, run

`th eval.lua -checkpoint_start_from model_id.t7 -loss_type logistic`

If you want to initialize with word2vec:
You should first run 
`python load_w2v.py`

Then during training, add option `-word2vec 1`.


