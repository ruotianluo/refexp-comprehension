require 'misc.RNN'
require 'nn'
require 'rnn'
require 'nngraph'

nn.FastLSTM.usenngraph = true

local SimNet, parent = torch.class('SimNet', 'nn.Module')

function SimNet:__init(opt)

  self.vocab_size = opt.vocab_size
  self.rnn_size = opt.rnn_size
  self.image_l1_size = opt.image_l1_size
  self.image_l2_size = opt.image_l2_size
  self.similarity = opt.similarity

  self.net = nn.Sequential()

  self.vis_net = nn.Sequential():add(nn.ParallelTable())
  self.vis_net:get(1):add(nn.Sequential()
    :add(nn.Linear(4096, self.image_l1_size))
    :add(nn.ReLU())
    :add(nn.BatchNormalization(self.image_l1_size)))
  self.vis_net:get(1):add(
    nn.Sequential():add(nn.Linear(4096, self.image_l1_size))
      :add(nn.ReLU())
      :add(nn.BatchNormalization(self.image_l1_size))
      :add(nn.Replicate(101))
      :add(nn.Squeeze()))
  self.vis_net:get(1):add(nn.BatchNormalization(8))
  self.vis_net:add(nn.JoinTable(1,1))
  self.vis_net:add(nn.Linear(self.image_l1_size*2+8,self.image_l2_size)):add(nn.ReLU()):add(nn.BatchNormalization(self.image_l2_size))

  self.language_net = nn.Sequential():add(nn.ParallelTable())
  self.language_net:get(1):add(nn.Sequencer(nn.LookupTableMaskZero(self.vocab_size + 2, self.rnn_size))):add(nn.Identity())
  self.language_net:add(BiDynamicRNN(nn.FastLSTM(self.rnn_size, self.rnn_size), nn.FastLSTM(self.rnn_size, self.rnn_size)))
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

function SimNet:parameters()
  return self.net:parameters()
end

function SimNet:clearState()
  self.net:clearState()
end

function SimNet:_build_sim_net()
  if self.similarity == 'dot' then
    assert(self.image_l2_size==self.rnn_size*2, 'The image feature and text feature has different dimension')
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local score = nn.Squeeze()(nn.MM(false, true)(inputs))
    return nn.gModule(inputs, {score})
  else
    local inputs = {}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local vis_feat = inputs[1]
    local text_feat = inputs[2]
    text_feat = nn.Replicate(101)(text_feat)
    feat = nn.JoinTable(1,1)({vis_feat, text_feat})
    local score = nn.Linear(self.image_l2_size+self.rnn_size*2, 1)(feat)
    score = nn.Squeeze()(nn.Tanh()(score))
    return nn.gModule(inputs, {score})
  end
end

function SimNet:updateOutput(inputs)
  return self.net:forward(inputs)
end

function SimNet:updateGradInput(inputs, gradOutput)
  return self.net:backward(inputs, gradOutput)
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

function struc_crit:updateGradInput(input, seq)
  return self.gradInput
end

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


local logistic_crit, parent = torch.class('LogisticCriterion', 'nn.Criterion')
function logistic_crit:__init()
  self.sigmoid = nn.Sigmoid()
  self.bce_crit = nn.BCECriterion()
  parent.__init(self)
end

function logistic_crit:updateOutput(input, iou)
  self.scores = self.sigmoid:forward(input)
  self.labels = torch.gt(iou, 0.5):typeAs(input)
  output = self.bce_crit:forward(self.scores, self.labels)
  return output
end

function logistic_crit:updateGradInput(input, iou)
  dscores = self.bce_crit:backward(self.scores, self.labels)
  self.gradInput = self.sigmoid:backward(input, dscores)
  return self.gradInput
end
