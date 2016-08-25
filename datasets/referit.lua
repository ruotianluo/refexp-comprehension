--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ReferIt dataset loader
--

local paths = require 'paths'
local py = require 'fb.python'
py.exec('import numpy as np')
py.exec('import cPickle as pkl')

local M = {}
local ReferItDataset = torch.class('ReferItDataset', M)

local __C = {}

__C.VAL_DATA_USE = 1000

__C.TRAIN_DATA_PAIRS = './data/train_pairs.pkl'
__C.VAL_DATA_PAIRS = './data/val_pairs.pkl'
__C.TEST_DATA_PAIRS = './data/test_pairs.pkl'
__C.CACHED_LOCAL_FEATURES_DIR = './data/referit_local_features/'
__C.CACHED_PROPOSAL_FEATURES_DIR = './data/referit_proposal_features/'
__C.CACHED_CONTEXT_FEATURES_DIR = './data/referit_context_features/'

function ReferItDataset:__init(opt, split)
  if split == 'train' then
    self.data_pairs = self:loadPklFile(__C.TRAIN_DATA_PAIRS)
  elseif split == 'val' then
    self.data_pairs = self:loadPklFile(__C.VAL_DATA_PAIRS)
  else
    self.data_pairs = self:loadPklFile(__C.TEST_DATA_PAIRS)
  end
  self.opt = opt
  self.split = split
  --self.dir = paths.concat(opt.data, split)
  --assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ReferItDataset:_get_iou(bbox_coordinate)
  all_boxes = bbox_coordinate
  gt_box = bbox_coordinate[1]

  all_areas = torch.cmul(bbox_coordinate[{{},7}], bbox_coordinate[{{},8}])
  gt_area = all_areas[1]

  x0 = torch.cmax(all_boxes[{{}, 1}],gt_box[1])
  y0 = torch.cmax(all_boxes[{{}, 2}],gt_box[2])
  x1 = torch.cmin(all_boxes[{{}, 3}],gt_box[3])
  y1 = torch.cmin(all_boxes[{{}, 4}],gt_box[4])

  w = torch.cmax(x1 - x0, 0)
  h = torch.cmax(y1 - y0, 0)

  intersection = torch.cmul(w, h)

  iou = torch.cdiv(intersection, all_areas + gt_area - intersection)

  return iou
end

function ReferItDataset:loadNpyFile(filename)
  return py.eval('np.load("'.. filename .. '")')
end
function ReferItDataset:loadNpzFile(filename)
  return py.eval('dict(np.load("'.. filename .. '"))')
end

function ReferItDataset:loadPklFile(filename)
  return py.eval('pkl.load(open("'.. filename .. '"))')
end

function ReferItDataset:get(i)
  blobs = {}
  imcrop_name, stream, bbox_feat, imname = unpack(self.data_pairs[i])

  T = 20
  vocab_size = 8800
  blobs['fc7_local'] = torch.zeros(101, 4096)
  blobs['fc7_context'] = torch.zeros(1, 4096)
  blobs['bbox_coordinate'] = torch.zeros(101, 8)
  blobs['sentence'] = torch.LongTensor(1, T):zero()
  blobs['sentence'][{1,1}]  = vocab_size + 1 -- bos

  stream = torch.Tensor(stream)
  if stream:size(1) > T-2 then
    stream = stream[{{1, T - 2}}]
  end
  blobs['length'] = torch.LongTensor({stream:size(1)+2})

  blobs['sentence'][{1, {2, 1+stream:size(1)}}]:copy(stream)
  blobs['sentence'][{1, 2+stream:size(1)}] = vocab_size + 2 -- eos
  
  blobs['fc7_local'][1] = self:loadNpyFile(paths.concat(__C.CACHED_LOCAL_FEATURES_DIR, imcrop_name .. '_fc7.npy'))
  blobs['bbox_coordinate'][1] = bbox_feat:squeeze()

  proposal_feats = self:loadNpzFile(paths.concat(__C.CACHED_PROPOSAL_FEATURES_DIR, imname .. '.npz'))
  if proposal_feats['local_feature']:size(1) >= 100 then
    blobs['fc7_local'][{{2, 101}, {}}] = proposal_feats['local_feature'][{{1, 100}, {}}]
    blobs['bbox_coordinate'][{{2, 101}, {}}] = proposal_feats['spatial_feat'][{{1, 100}, {}}]
  elseif proposal_feats['local_feature']:size(1) < 100 then
    print("proposal number less than 100")
    p_len = proposal_feats['local_feature']:size(1)
    blobs['fc7_local'][{{2, 2+p_len-1}}]:copy(proposal_feats['local_feature'])
    blobs['bbox_coordinate'][{{2, 2+p_len-1}}]:copy(proposal_feats['spatial_feat'])
    for k = 2,  101 - p_len do
      blobs['fc7_local'][k + p_len] = proposal_feats['local_feature'][1]
      blobs['bbox_coordinate'][k + p_len] = proposal_feats['spatial_feat'][1]
    end
  end

  blobs['fc7_context'][1] = self:loadNpyFile(paths.concat(__C.CACHED_CONTEXT_FEATURES_DIR, imname .. '_fc7.npy'))

  --calculate iou
  blobs['iou'] = self:_get_iou(blobs['bbox_coordinate'])

  assert(blobs['iou']:dim() == 1)

  return blobs
end

function ReferItDataset:size()
  return #self.data_pairs
end

return M.ReferItDataset