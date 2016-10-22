--[[
Main entry point for training a DenseCap model
]]--

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'nngraph'
require 'misc.optim_updates'
require 'nn'
require 'misc.SimNet2'
require 'misc.Reader'
require 'misc.BOW_MLP'

local DataLoader = require 'dataloader'
local utils = require 'misc.utils'
local opts = require 'train_opts'

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpuid >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-- initialize the data loader class
testLoader = DataLoader.create(opt, {'test'})
opt.vocab_size = 8800
--opt.idx_to_token = loader.info.idx_to_token

-- Initialize training information
local loss_history = {}
local lr_history = {}
local val_loss_history = {}
local val_results_history = {}
local iter = 1
local epoch = 1
local optim_state = {}
local best_val_score

if string.len(opt.checkpoint_start_from) > 0 then
  -- load protos from file
  print('initializing training information from ' .. opt.checkpoint_start_from)
  local loaded_checkpoint = torch.load(opt.checkpoint_start_from)
  protos = loaded_checkpoint.protos

  if opt.loss_type == 'structure' then
    protos.crit = SturctureCriterion(opt)
  else
    protos.crit = SoftmaxCriterion(opt)
  end

  iter = loaded_checkpoint.iter + 1 or iter
  loss_history = loaded_checkpoint.loss_history or loss_history
  lr_history = loaded_checkpoint.lr_history or lr_history
  val_loss_history = loaded_checkpoint.val_loss_history or val_loss_history
  val_results_history = loaded_checkpoint.val_results_history or val_results_history
  optim_state = loaded_checkpoint.optim_state or optim_state
  if opt.load_best_score == 1 then
    best_val_score = loaded_checkpoint.best_val_score
  end
else
  -- create protos from scratch
  -- intialize language model
  protos = {}
  protos.net = SimNet(opt)
  
  if opt.loss_type == 'structure' then
    protos.crit = SturctureCriterion(opt)
  else
    protos.crit = SoftmaxCriterion(opt)
  end
end

if opt.gpuid >= 0 then
  protos.net:cuda()
  protos.crit:cuda()
end

if cudnn then
  cudnn.convert(protos.net, cudnn)
  --cudnn.convert(protos.crit, cudnn)
end

-- get the parameters vector
local params, grad_params = protos.net:getParameters()
assert(params:nElement() == grad_params:nElement(), 'number of parameters doesn\'t match')
print('total number of parameters in net: ', grad_params:nElement())

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(eval_set, evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = -1

  protos.net:evaluate()

  local loss_sum = 0
  local loss_evals = 0
  local K = 100
  local topK_correct_num = torch.zeros(K)
  local total_num = 0

  for n, data in eval_set:run() do

    if opt.gpuid >= 0 then
      for k,v in pairs(data) do
        if torch.type(data[k]) ~= 'torch.LongTensor' then
          data[k] = data[k]:cuda()
        else
          data[k] =  data[k]:cuda()
        end
      end
    end

    -- fetch a batch of data
    local inputs = {{data.fc7_local, data.fc7_context, data.bbox_coordinate},{data.sentence, data.length}}
    local scores = protos.net:forward(inputs)
    local loss = protos.crit:forward(scores, data.iou)

    scores = scores[{{2,-1}}]:float()
    local IoUs = data['iou'][{{2, -1}}]:float()
    -- Evaluate the correctness of top K predictions
    _, topK_ids = torch.sort(-scores)
    topK_IoUs = IoUs:index(1, topK_ids[{{1, K}}])
    -- whether the K-th (ranking from high to low) candidate is correct
    topK_is_correct = torch.ge(topK_IoUs, 0.5):float()
    -- whether at least one of the top K candidates is correct
    topK_any_correct = torch.gt(torch.cumsum(topK_is_correct), 0)
    topK_correct_num:add(topK_any_correct:float())
    total_num = total_num + 1

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
        
    if verbose then
      print(string.format('evaluating validation preformance... %d/%d (%f)', n, val_images_use, loss))
    end

    if n % 1000 == 0 then
      print(topK_correct_num[1]/total_num, topK_correct_num[10]/total_num, topK_correct_num[100]/total_num)
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if val_images_use ~= -1 and n >= val_images_use then break end -- we've used enough images
  end

  local val_result = {recall_1 = topK_correct_num[1]/total_num, recall_10 = topK_correct_num[10]/total_num}
  print(val_result)
  eval_set:resetThreads()

  return loss_sum/loss_evals, val_result
end


-- evaluate the validation performance
local loss, result = eval_split(testLoader, {})
print('validation loss: ', val_loss)
print(val_result)
utils.write_json(path.join('result', 'result_' .. opt.id .. '.json'), {loss = loss, result = result})