require 'rnn'
require 'misc.RNN'
local gradcheck = require 'misc.gradcheck'

torch.manualSeed(0)
math.randomseed(0)

-- hyper-parameters 
batchSize = 8
rho = 10 -- sequence length
hiddenSize = 5
nIndex = 10
lr = 0.1
maxIter = 100

sharedLookupTable = nn.LookupTableMaskZero(nIndex, hiddenSize)

brnn = BiDynamicRNN(nn.FastLSTM(hiddenSize, hiddenSize), nn.FastLSTM(hiddenSize, hiddenSize), true)

parallel = nn.ParallelTable()
parallel:add(nn.Sequencer(sharedLookupTable)):add(nn.Identity())

net = nn.Sequential():add(parallel):add(brnn)

sequence_ = torch.LongTensor():range(1,10) -- 1,2,3,4,5,6,7,8,9,10
sequence = torch.LongTensor(100,10):copy(sequence_:view(1,10):expand(100,10))
sequence:resize(100*10) -- one long sequence of 1,2,3...,10,1,2,3...10...

offsets = {}
maxStep = {}
for i=1,batchSize do
   table.insert(offsets, math.ceil(math.random()*sequence:size(1)))
   -- variable length for each sample
   table.insert(maxStep, math.random(rho))
end
offsets = torch.LongTensor(offsets)

inputs = torch.zeros(rho, batchSize)
inputs_rev = torch.zeros(rho, batchSize)
for step=1,rho do
  -- a batch of inputs
  inputs[step] = sequence:index(1, offsets)
  -- increment indices
  offsets:add(1)
  for j=1,batchSize do
      if offsets[j] > sequence:size(1) then
        offsets[j] = 1
      end
  end
  -- padding
  for j=1,batchSize do
    if step > maxStep[j] then
        inputs[step][j] = 0
    end
  end
end

seq_length = torch.Tensor(maxStep)

gt = torch.randn(8, 10)

crit = nn.MSECriterion()

local output = net:forward({inputs, seq_length})
local loss = crit:forward(output, gt)
local gradOutput = crit:backward(output, gt)
local gradInput, dummy = unpack(net:backward({inputs, seq_length}, gradOutput))

-- create a loss function wrapper
local function f(x)
  local output = net:forward{x, seq_length}
  local loss = crit:forward(output, )
  return loss
end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)

  -- print(gradInput)
  -- print(gradInput_num)
  -- local g = gradInput:view(-1)
  -- local gn = gradInput_num:view(-1)
  -- for i=1,g:nElement() do
  --   local r = gradcheck.relative_error(g[i],gn[i])
  --   print(i, g[i], gn[i], r)
  -- end

  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 5e-4)

