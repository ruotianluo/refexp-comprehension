require 'rnn'
require 'misc.RNN'
require 'misc.CNN'
local gradcheck = require 'misc.gradcheck'

torch.manualSeed(0)
math.randomseed(0)

-- hyper-parameters 
batchSize = 8
rho = 21 -- sequence length
hiddenSize = 10
nIndex = 10
lr = 0.1
maxIter = 100

sharedLookupTable = nn.LookupTableMaskZero(nIndex, hiddenSize)

--wcnn = WordCNN(hiddenSize, hiddenSize)
wcnn = WordHybridCNN(hiddenSize, hiddenSize, nil, {})

parallel = nn.ParallelTable()
parallel:add(nn.Sequencer(sharedLookupTable)):add(nn.Identity())

net = nn.Sequential():add(parallel):add(wcnn)


-- Get data
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

-- Make it batch_size * rho
inputs = inputs:transpose(1,2)

local output = net:forward({inputs, seq_length})
local loss = crit:forward(output, gt)
local gradOutput = crit:backward(output, gt)
local gradInput, dummy = unpack(net:backward({inputs, seq_length}, gradOutput))

print(output)




