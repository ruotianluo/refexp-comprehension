require 'misc.RNN'
require 'nn'
require 'rnn'
require 'cbp'
local net_utils = require 'misc.net_utils'
require 'nngraph'

nn.FastLSTM.usenngraph = true

local Reader, parent = torch.class('Reader', 'nn.Module')

function Reader:__init(opt)

  self.vocab_size = opt.vocab_size
  self.rnn_size = opt.rnn_size
  self.image_l1_size = opt.image_l1_size
  self.image_l2_size = opt.image_l2_size
  self.score_type = opt.score_type

  if opt.visnet_type == 'rt' then
    self.vis_net = net_utils.build_visnet_rt(opt)
  elseif opt.visnet_type == 'lc' then
    self.vis_net = net_utils.build_visnet_lc(opt)
  elseif opt.visnet_type == 'old' then
    self.vis_net = net_utils.build_visnet_old(opt)  
  end

  if opt.normalize == 1 then
    self.vis_net:add(nn.Normalize(2))
  end

  self.embed_net = nn.Sequencer(nn.LookupTableMaskZero(self.vocab_size + 2, self.rnn_size))

  self.joint_net = nn.Sequential()
  self.joint_net:add(BiDynamicRNN(nn.FastLSTM(self.image_l2_size + self.rnn_size, self.rnn_size), nn.FastLSTM(self.image_l2_size + self.rnn_size, self.rnn_size), nil, opt.state_type))
  self.joint_net:add(nn.Linear(self.rnn_size * 2, self.rnn_size))
  self.joint_net:add(nn.ReLU())
  self.joint_net:add(nn.Linear(self.rnn_size, 1))
  self.joint_net:add(nn.Squeeze())

  if self.score_type == 'cbp' then
    self.combine_net = nn.Sequential()
    self.combine_net:add(nn.CompactBilinearPooling(self.image_l2_size + self.rnn_size))
      :add(nn.SignedSquareRoot())
      :add(nn.Normalize(2))
  end

  self.net = {self.vis_net, self.embed_net, self.joint_net}
end

function Reader:training()
  parent.training(self)
  for k,v in pairs(self.net) do
    v:training()
  end
end

function Reader:evaluate()
  parent.evaluate(self)
  for k,v in pairs(self.net) do
    v:evaluate()
  end
end

function Reader:parameters()
  local params = {}
  local grad_params = {}
  for k,v in pairs(self.net) do
    local p, g = v:parameters()
    for _k,_v in pairs(p) do table.insert(params, _v) end
    for _k,_v in pairs(g) do table.insert(grad_params, _v) end
  end
  return params, grad_params
end

function Reader:clearState()
  for k,v in pairs(self.net) do
    v:clearState()
  end
  return parent.clearState(self)
end

function Reader:updateOutput(inputs)
  -- inputs: {{data.fc7_local, data.fc7_context, data.bbox_coordinate},{data.sentence, data.length}}
  self.vis_feats = self.vis_net:forward(inputs[1])
  self.word_embed = self.embed_net:forward(inputs[2][1])
  self.vis_feats = self.vis_feats:view(self.vis_feats:size(1), 1, self.vis_feats:size(2))
    :expand(self.vis_feats:size(1), self.word_embed:size(2), self.vis_feats:size(2))
  self.word_embed = self.word_embed:expand(self.vis_feats:size(1), self.word_embed:size(2), self.word_embed:size(3))

  if self.score_type == 'cbp' then
    self.vis_feats = self.vis_feats:contiguous()
    self.word_embed = self.word_embed:contiguous()
    self.joint_input = self.combine_net:forward(
      {self.vis_feats:view(-1, self.image_l2_size), self.word_embed:view(-1, self.image_l2_size)})
    self.joint_input = self.joint_input:view(self.vis_feats:size(1), self.word_embed:size(2), self.image_l2_size + self.rnn_size)
  else
    self.joint_input = torch.cat({self.vis_feats, self.word_embed}, 3)
  end

  self.output = self.joint_net:forward({self.joint_input, inputs[2][2]:expand(self.vis_feats:size(1))})
  return self.output
end

function Reader:updateGradInput(inputs, gradOutput)
  local djoint_input = self.joint_net:backward({self.joint_input, inputs[2][2]:expand(self.vis_feats:size(1))}, gradOutput)
  djoint_input = djoint_input[1]
  local dvis_feats
  local dword_embed
  if self.score_type == 'cbp' then
    dvis_feats, dword_embed = unpack(self.combine_net:backward(
      {self.vis_feats:view(-1, self.image_l2_size), self.word_embed:view(-1, self.image_l2_size)},
      djoint_input:view(-1, self.image_l2_size + self.rnn_size)))
    dvis_feats = dvis_feats:view(self.vis_feats:size(1), self.word_embed:size(2), self.image_l2_size)
    dword_embed = dword_embed:view(self.vis_feats:size(1), self.word_embed:size(2), self.rnn_size)
  else
    dvis_feats = djoint_input[{{},{},{1, self.vis_feats:size(2)}}]
    dword_embed = djoint_input[{{},{},{self.vis_feats:size(2) + 1, -1}}]
  end
  dvis_feats = dvis_feats:sum(2):squeeze(2)
  dword_embed = dword_embed:sum(1)
  
  self.embed_net:backward(inputs[2][1], dword_embed)
  self.gradInput = {}
  self.gradInput[1] = self.vis_net:backward(inputs[1], dvis_feats)
  self.gradInput[2] = {}
  return self.gradInput
end

function Reader:clearState()
  self.vis_feats:set()
  self.word_embed:set()
  self.joint_input:set()
  self.gradInput = {}
end
