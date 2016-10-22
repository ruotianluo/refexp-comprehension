require 'misc.RNN2'
require 'nn'
require 'rnn'
require 'cbp'
require 'dpnn'
local net_utils = require 'misc.net_utils'
require 'nngraph'
require 'misc.CNN'

nn.FastLSTM.usenngraph = true

local BOW, parent = torch.class('nn.BOW', 'nn.Module')
function BOW:__init()
end

function BOW:parameters()
  return 'can\'t be used' -- To make sure the dpnn getParameter won't visit this
end

function BOW:updateOutput(inputs)
  local vectors, sequence_length = inputs[1], inputs[2]
  self.output = torch.sum(vectors, 2):squeeze(2)
  self.output:cdiv(sequence_length:typeAs(self.output):view(-1, 1):expandAs(self.output))
  return self.output
end

function BOW:updateGradInput(inputs, gradOutput)
  self.gradInput = torch.cdiv(gradOutput, sequence_length:typeAs(gradOutput):view(-1, 1):expandAs(gradOutput))
  self.gradInput = {self.gradInput:view(inputs[1]:size(1), 1, inputs[1]:size(3)):expandAs(inputs[1]), nil}
  return self.gradInput  
end

local BOW_MLP, parent = torch.class('BOW_MLP', 'nn.Module')

function BOW_MLP:__init(opt)

  self.vocab_size = opt.vocab_size
  self.score_type = opt.score_type -- 'concat == lienar, Bilinear == dot, '

  self.net = self:_build_net()

  self.language_net = nn.Sequential():add(nn.ParallelTable())

  self.lookup = nn.LookupTableMaskZero(self.vocab_size + 2, 300)
  self:loadW2V(self.lookup)

  self.language_net:get(1):add(self.lookup):add(nn.Identity())
  self.language_net:add(nn.BOW())

  self.language_net.dpnn_getParameters_found = true -- Prevent from finetuning

end

function BOW_MLP:loadW2V(lookup)
  require 'hdf5'
  w2v = hdf5.open('data/w2v.h5', 'r')
  w2v = w2v:read('/w2v'):all()
  lookup.weight[{{2,-1}}]:copy(w2v)
end

function BOW_MLP:setBatchSize(batch_size)
  for k, v in pairs(self.net:findModules('nn.Replicate')) do
    v.nfeatures = batch_size
  end
end

function BOW_MLP:training()
  parent.training(self)
  self.net:training()
end

function BOW_MLP:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end

function BOW_MLP:clearState()
  self.net:clearState()
  return parent.clearState(self)
end

function BOW_MLP:_build_net()
  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local fc7_local = nn.Normalize(2)( inputs[1])
  local bbox_feats = nn.Normalize(2)(inputs[2])
  local sent_feats = nn.Normalize(2)(inputs[3])

  local img_feats = nn.JoinTable(1, 1)({fc7_local, bbox_feats})

  if self.score_type == 'dot' then
    local proj_img = nn.Linear(4096 + 8, 300)(img_feats)
    local score = nn.Squeeze(2)(nn.MM(false, true)({proj_img, sent_feats}))
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})
  elseif self.score_type == 'concat' then
    img_feats = nn.Linear(4096 + 8, 1)(img_feats)
    sent_feats = nn.Linear(300, 1)(sent_feats)
    sent_feats = nn.Squeeze(1,2)(nn.Replicate(101)(sent_feats))
    local score = nn.CAddTable()({img_feats, sent_feats})
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})  
  elseif self.score_type == 'mlp' then
    img_feats = nn.Linear(4096 + 8, 1024)(img_feats)
    sent_feats = nn.Linear(300, 1024)(sent_feats)
    sent_feats = nn.Squeeze(1,2)(nn.Replicate(101)(sent_feats))
    -- sent_feats = nn.PrintSize('sent_feats')(sent_feats)
    -- img_feats = nn.PrintSize('img_feats')(img_feats)
    local feat = nn.Dropout()(nn.CAddTable()({img_feats, sent_feats}))
    local score = nn.Squeeze(2)(nn.Linear(1024, 1)(feat))
    score = nn.Squeeze()(score)
    return nn.gModule(inputs, {score})  
  end
end

function BOW_MLP:updateOutput(inputs)
  self:setBatchSize(inputs[1][1]:size(1))
  self.sent_feats = self.language_net:forward(inputs[2])
  self.output = self.net:forward({inputs[1][1], inputs[1][3], self.sent_feats})
  return self.output
end

function BOW_MLP:updateGradInput(inputs, gradOutput)
  self.gradInput = {self.net:backward({inputs[1][1], inputs[1][3], self.sent_feats}, gradOutput), {nil, nil}}
  return self.gradInput
end