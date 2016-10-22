require 'misc.RNN2'
require 'nn'
require 'rnn'
require 'cbp'
local net_utils = require 'misc.net_utils'
require 'nngraph'
require 'misc.CNN'

nn.FastLSTM.usenngraph = true

local SimNet, parent = torch.class('SimNet', 'nn.Module')

function SimNet:__init(opt)

  self.vocab_size = opt.vocab_size
  self.rnn_size = opt.rnn_size
  self.word_embedding_size = opt.word_embedding_size
  self.image_l1_size = opt.image_l1_size
  self.image_l2_size = opt.image_l2_size
  self.score_type = opt.score_type

  self.net = nn.Sequential()

  --[[self.vis_net = nn.Sequential():add(nn.ParallelTable())
  self.vis_net:get(1):add(nn.Sequential()
    :add(nn.Linear(4096, self.image_l1_size))
    :add(nn.ReLU()))
    --:add(nn.BatchNormalization(self.image_l1_size)))
  self.vis_net:get(1):add(
    nn.Sequential():add(nn.Linear(4096, self.image_l1_size))
      :add(nn.ReLU())
      --:add(nn.BatchNormalization(self.image_l1_size))
      :add(nn.Replicate(101))
      :add(nn.Squeeze(2)))
  
  self.vis_net:get(1):add(nn.BatchNormalization(8))
  self.vis_net:add(nn.JoinTable(1,1))
  self.vis_net
    :add(nn.Linear(self.image_l1_size*2+8,self.image_l2_size))
    :add(nn.ReLU())
    :add(nn.BatchNormalization(self.image_l2_size))
  ]]--
  if opt.visnet_type == 'rt' then
    self.vis_net = net_utils.build_visnet_rt(opt)
  elseif opt.visnet_type == 'lc' then
    self.vis_net = net_utils.build_visnet_lc(opt)
  elseif opt.visnet_type == 'old' then
    self.vis_net = net_utils.build_visnet_old(opt)  
  end

  self.language_net = nn.Sequential():add(nn.ParallelTable())
  if opt.word2vec == 1 then
    self.word_embedding_size = 300
    self.lookup = nn.LookupTableMaskZero(self.vocab_size + 2, self.word_embedding_size)
    self:loadW2V(self.lookup)
  else
    self.lookup = nn.LookupTableMaskZero(self.vocab_size + 2, self.word_embedding_size)
  end

  self.language_net:get(1):add(nn.Sequencer(self.lookup)):add(nn.Identity())
  if opt.lang_embed_type == 'rnn' then
    self.language_net:add(nn.DynamicRNN(nn.FastLSTM, {state_type = opt.state_type, input_size = self.word_embedding_size, hidden_size = self.rnn_size, layer_num = opt.layer_num}))
  elseif opt.lang_embed_type == 'cnn' then
    self.language_net:add(WordCNN(self.word_embedding_size, self.rnn_size * 2))
  elseif opt.lang_embed_type == 'cnnhybrid' then
    self.language_net:add(WordHybridCNN(self.word_embedding_size, self.rnn_size * 2, nil, opt))
  end

  -- self.language_net:add(BiDynamicRNN(nn.FastLSTM(self.rnn_size, self.rnn_size), nn.FastLSTM(self.rnn_size, self.rnn_size), nil, opt.state_type))
  -- self.language_net:add(nn.BatchNormalization(self.rnn_size * 2))

  if opt.normalize == 1 then
    self.vis_net:add(nn.Normalize(2))
    self.language_net:add(nn.Normalize(2))
  end

  self.sim_net = self._build_sim_net(opt)

  self.net:add(nn.ParallelTable())
  self.net:get(1):add(self.vis_net):add(self.language_net)

  self.net:add(self.sim_net)
end

function SimNet:loadW2V(lookup)
  require 'hdf5'
  w2v = hdf5.open('data/w2v.h5', 'r')
  w2v = w2v:read('/w2v'):all()
  lookup.weight[{{2,-1}}]:copy(w2v)
end

function SimNet:setBatchSize(batch_size)
  for k, v in pairs(self.net:findModules('nn.Replicate')) do
    v.nfeatures = batch_size
  end
end

function SimNet:training()
  parent.training(self)
  self.net:training()
end

function SimNet:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end

function SimNet:parameters()
  return self.net:parameters()
end

function SimNet:clearState()
  self.net:clearState()
  return parent.clearState(self)
end

function SimNet:_build_sim_net()
  if self.score_type == 'dot' then
    assert(self.image_l2_size==self.rnn_size*2, 'The image feature and text feature has different dimension')
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local score = nn.Squeeze(2)(nn.MM(false, true)(inputs))
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})
  elseif self.score_type == 'concat' then
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local vis_feat = inputs[1]
    local text_feat = inputs[2]
    vis_feat = nn.Linear(self.image_l2_size, self.rnn_size*2)(vis_feat)
    text_feat = nn.Linear(self.rnn_size*2, self.rnn_size*2)(text_feat)
    text_feat = nn.Squeeze(1,2)(nn.Replicate(101)(text_feat))
    local feat = nn.ReLU()(nn.CAddTable()({vis_feat, text_feat}))
    local score = nn.Squeeze(2)(nn.Linear(self.rnn_size*2, 1)(feat))
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})  
  elseif self.score_type == 'euclidean' or self.score_type == 'cosine' or self.score_type == 'cbp' or self.score_type == 'ord_embed' then
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local vis_feat = inputs[1]
    local text_feat = inputs[2]
    text_feat = nn.Squeeze()(nn.Replicate(101)(text_feat))
    local score
    if self.score_type == 'euclidean' then
      score = nn.MulConstant(-1)(nn.PairwiseDistance(2)({vis_feat, text_feat}))
    elseif self.score_type == 'cosine' then
      score = nn.CosineDistance()({vis_feat, text_feat})
    elseif self.score_type == 'cbp' then
      local score_net = nn.Sequential()
      score_net:add(nn.CompactBilinearPooling(self.image_l2_size))
        :add(nn.SignedSquareRoot())
        :add(nn.Normalize(2))
        :add(nn.Linear(self.image_l2_size, 1))
      score = score_net({vis_feat, text_feat})
    elseif self.score_type == 'ord_embed' then
      text_feat = nn.Normalize(2)(nn.Abs()(text_feat))
      vis_feat = nn.Normalize(2)(nn.Abs()(vis_feat))
      score = nn.CSubTable()({text_feat,vis_feat})
      score = nn.ReLU()(score)
      score = nn.Square()(score)
      score = nn.Sum(1,1)(score)
    end
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})
  end
end

function SimNet:updateOutput(inputs)
  self:setBatchSize(inputs[1][1]:size(1))
  self.output = self.net:forward(inputs)
  return self.output
end

function SimNet:updateGradInput(inputs, gradOutput)
  self.gradInput = self.net:backward(inputs, gradOutput)
  return self.gradInput
end

local struc_crit, parent = torch.class('SturctureCriterion', 'nn.Criterion')
function struc_crit:__init(opt)
  self.slack_rescaled = opt.slack_rescaled
  parent.__init(self)
end

function struc_crit:updateOutput(input, iou)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local diff
  if self.slack_rescaled == 1 then
    diff = torch.cmul(1 - iou, 1 + input - input[1])
  else
    diff = (-iou + 1) + input - input[1]
  end
  local output, idx = torch.max(diff, 1)
  idx = idx[1] -- Turn LongTensor to number
  output = output[1]
  
  self.gradInput[1] = - 1
  self.gradInput[idx] = 1
  if self.slack_rescaled == 1 then
    self.gradInput[1] = (1 - iou[idx]) * self.gradInput[1]
    self.gradInput[idx] = (1 - iou[idx]) * self.gradInput[idx]
  end

  return output
end

function struc_crit:updateGradInput(input, iou)
  return self.gradInput
end

local hinge_crit, parent = torch.class('HingeCriterion', 'nn.Criterion')
function hinge_crit:__init(opt)
  self.margin = opt.margin
  parent.__init(self)
end

function hinge_crit:updateOutput(input, iou)
  -- margin + max(negative) - min(gt)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local gt_idx = torch.ge(iou, 0.5):float():nonzero():view(-1)
  local min_v, min_idx = torch.min(input:index(1, gt_idx), 1)
  min_v = min_v[1]
  min_idx = gt_idx[min_idx[1]]

  local neg_idx = torch.lt(iou, 0.5):float():nonzero():view(-1)
  local max_v, max_idx = torch.max(input:index(1, neg_idx) , 1)
  max_v = max_v[1]
  max_idx = neg_idx[max_idx[1]]

  if self.margin + max_v - min_v <= 0 then
    return 0
  else
    self.gradInput[min_idx] = -1
    self.gradInput[max_idx] = 1
    return self.margin + max_v - min_v
  end
end

function hinge_crit:updateGradInput(input, iou)
  return self.gradInput
end

-- weighted
local softmax_crit, parent = torch.class('SoftmaxCriterion', 'nn.Criterion')
function softmax_crit:__init()
  self.log_softmax =  nn.LogSoftMax()
  parent.__init(self)
end

function softmax_crit:updateOutput(input, iou)
  local log_prob = self.log_softmax:forward(input)
  local weight = torch.cmul(iou, torch.gt(iou,0.5):typeAs(iou))
  local output = - torch.sum(torch.cmul(log_prob, weight))
  self.gradInput = self.log_softmax:backward(input, - weight)
  return output
end

function softmax_crit:updateGradInput(input, iou)
  local weight = torch.cmul(iou, torch.gt(iou,0.5):typeAs(iou))
  self.gradInput = self.log_softmax:backward(input, - weight)
  return self.gradInput
end

local softmax2_crit, parent = torch.class('Softmax2Criterion', 'nn.Criterion')
function softmax2_crit:__init()
  self.log_softmax =  nn.LogSoftMax()
  parent.__init(self)
end

function softmax2_crit:updateOutput(input, iou)
  local log_prob = self.log_softmax:forward(input)
  local weight = torch.gt(iou,0.5):typeAs(iou)
  local output = - torch.sum(torch.cmul(log_prob, weight))
  self.gradInput = self.log_softmax:backward(input, - weight)
  return output
end

function softmax2_crit:updateGradInput(input, iou)
  local weight = torch.gt(iou,0.5):typeAs(iou)
  self.gradInput = self.log_softmax:backward(input, - weight)
  return self.gradInput
end


local logistic_crit, parent = torch.class('LogisticCriterion', 'nn.Criterion')
function logistic_crit:__init()
  self.sigmoid = nn.Sigmoid()
  self.bce_crit = nn.BCECriterion()
  parent.__init(self)
end

function logistic_crit:updateOutput(input, iou)
  self.scores = self.sigmoid:forward(input)
  self.labels = torch.gt(iou, 0.5):typeAs(input)
  local output = self.bce_crit:forward(self.scores, self.labels)
  return output
end

function logistic_crit:updateGradInput(input, iou)
  local dscores = self.bce_crit:backward(self.scores, self.labels)
  self.gradInput = self.sigmoid:backward(input, dscores)
  return self.gradInput
end
