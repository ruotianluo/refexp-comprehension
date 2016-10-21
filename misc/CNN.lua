require 'nn'
require 'misc.RNN'

local WordCNN, parent = torch.class('WordCNN', 'nn.Module')

function WordCNN:__init(input_size, output_size, length, opt)
   -- if length == 10 then

  local net = nn.Sequential()
  net:add(nn.Transpose({2,3}))
  net:add(nn.Unsqueeze(4))
  -- 11 x alphasize
  net:add(nn.SpatialConvolution(input_size, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialMaxPooling(1, 3, 1, 2, 0, 0))
  -- 5 x 256
  net:add(nn.SpatialConvolution(512, 1024, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))
  
  net:add(nn.SpatialConvolution(1024, 1024, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))

  net:add(nn.SpatialMaxPooling(1, 5, 1, 5, 0, 0))
  -- 1 x 512
  net:add(nn.Reshape(1024))
  net:add(nn.Linear(1024, output_size))

  self.net = net
end

function WordCNN:parameters()
  return self.net:parameters()
end

function WordCNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  self.output = self.net:forward(inputs[1])
  return self.output
end

function WordCNN:updateGradInput(inputs, gradOutput)
  self.gradInput = {self.net:backward(inputs[1], gradOutput), nil}
  return self.gradInput
end

local CharCNN, parent = torch.class('CharCNN', 'nn.Module')

function CharCNN:__init(input_size, output_size, length, opt)
  --if length == 10 then

  local net = nn.Sequential()
  net:add(nn.Transpose({2,3}))
  net:add(nn.Unsqueeze(4))
  -- 61 x alphasize
  net:add(nn.SpatialConvolution(input_size, 256, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialConvolution(256, 256, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialMaxPooling(1, 3, 1, 3, 0, 0))
  -- 20 x 256
  net:add(nn.SpatialConvolution(256, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))
  
  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))

  net:add(nn.SpatialMaxPooling(1, 3, 1, 3, 0, 0))

  -- 6 * 512
  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))
  
  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))

  --net:add(nn.SpatialMaxPooling(1, 4, 1, 4, 0, 0))
  -- 1 x 512
  net:add(nn.Reshape(512 * 6))
  net:add(nn.Dropout(0.5))
  net:add(nn.Linear(512 * 6, output_size))

  self.net = net
end

function CharCNN:parameters()
  return self.net:parameters()
end

function CharCNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  self.output = self.net:forward(inputs[1])
  return self.output
end

function CharCNN:updateGradInput(inputs, gradOutput)
  self.gradInput = {self.net:backward(inputs[1], gradOutput), nil}
  return self.gradInput
end



local WordHybridCNN, parent = torch.class('WordHybridCNN', 'nn.Module')
function WordHybridCNN:__init(input_size, output_size, length, opt)
  self.state_type = opt.state_type or 'final'

  local net = nn.Sequential()
  net:add(nn.Transpose({2,3}))
  net:add(nn.Unsqueeze(4))
  -- 11 x alphasize
  net:add(nn.SpatialConvolution(input_size, 1025, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialConvolution(1024, 1024, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))
  
  -- b * 512 * 5 * 1
  net:add(nn.Squeeze(4))
  net:add(nn.Transpose({2,3}))
  -- b * 5 * 512 
  net = nn.Sequential():add(nn.ParallelTable():add(net):add(nn.Identity()))
  net:add(BiDynamicRNN(nn.FastLSTM(1024, output_size), nn.FastLSTM(1024, output_size), nil, opt.state_type))
  net:add(nn.Linear(2 * output_size, output_size))

  self.net = net
end

function WordHybridCNN:parameters()
  return self.net:parameters()
end

function WordHybridCNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  self.output = self.net:forward(inputs)

  return self.output
end

function WordHybridCNN:updateGradInput(inputs, gradOutput)
  local dinput = self.net:backward(inputs, gradOutput)
  self.gradInput = {dinput[1], nil}
  return self.gradInput
end

local CharHybridCNN, parent = torch.class('CharHybridCNN', 'nn.Module')
function CharHybridCNN:__init(input_size, output_size, length, opt)

  local net = nn.Sequential()
  net:add(nn.Transpose({2,3}))
  net:add(nn.Unsqueeze(4))
  -- 61 x alphasize
  net:add(nn.SpatialConvolution(input_size, 256, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(256))

  net:add(nn.SpatialConvolution(256, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))

  net:add(nn.SpatialMaxPooling(1, 3, 1, 3, 0, 0))

  -- 20
  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))

  net:add(nn.SpatialConvolution(512, 512, 1, 3, 1, 1, 0, 1))
  net:add(nn.ReLU())
  -- net:add(nn.SpatialBatchNormalization(512))
  
  -- b * 512 * 20 * 1
  net:add(nn.Squeeze(4))
  net:add(nn.Transpose({2,3}))
  -- b * 20 * 512 
  net = nn.Sequential():add(nn.ParallelTable():add(net):add(nn.Identity()))
  net:add(BiDynamicRNN(nn.FastLSTM(512, output_size), nn.FastLSTM(512, output_size), nil, opt.state_type))
  net:add(nn.Linear(2 * output_size, output_size))

  self.net = net
end

function CharHybridCNN:parameters()
  return self.net:parameters()
end

function CharHybridCNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  self.output = self.net:forward({inputs[1], torch.floor((inputs[2] - 3) / 3 + 1)})
  return self.output
end

function CharHybridCNN:updateGradInput(inputs, gradOutput)
  local dinput = self.net:backward({inputs[1], torch.floor((inputs[2] - 3) / 3 + 1)})
  self.gradInput = {dinput[1], nil}
  return self.gradInput
end


