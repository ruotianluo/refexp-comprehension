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
require 'misc.SimNet'

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
local trainLoader, valLoader = DataLoader.create(opt, {'train', 'val'})
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

function build_crit(opt)
  if opt.loss_type == 'structure' then
    return SturctureCriterion(opt)
  elseif opt.loss_type == 'softmax' then
    return SoftmaxCriterion()
  elseif opt.loss_type == 'logistic' then
    return LogisticCriterion()
  elseif opt.loss_type == 'hinge' then
    return HingeCriterion(opt)
  end
end

if string.len(opt.checkpoint_start_from) > 0 then
  -- load protos from file
  print('initializing training information from ' .. opt.checkpoint_start_from)
  local loaded_checkpoint = torch.load(opt.checkpoint_start_from)
  protos = loaded_checkpoint.protos
  protos.crit = build_crit(opt)
  
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
  protos.crit = build_crit(opt)
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
  local val_images_use = utils.getopt(evalopt, 'val_images_use', -1)

  protos.net:evaluate()

  local loss_sum = 0
  local loss_evals = 0
  local K = 100
  local topK_correct_num = torch.zeros(K)
  local topK_correct_boxes = 0
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
    topK_correct_boxes = topK_correct_boxes + torch.sum(topK_is_correct)
    topK_correct_num:add(topK_any_correct:float())
    total_num = total_num + 1

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
        
    if verbose then
      print(string.format('evaluating validation preformance... %d/%d (%f)', n, val_images_use, loss))
    end

    if loss_evals % 10 == 0 then collectgarbage() end
    if val_images_use ~= -1 and n >= val_images_use then break end -- we've used enough images
  end

  local val_result = {recall_1 = topK_correct_num[1]/total_num, recall_10 = topK_correct_num[10]/total_num}
  print(val_result)
  print('Average true proposals:')
  print(topK_correct_boxes/total_num)
  eval_set:resetThreads()

  return loss_sum/loss_evals, val_result
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun(data)
  grad_params:zero()
  protos.net:training()
  if opt.gpuid >= 0 then
    for k,v in pairs(data) do
      if torch.type(data[k]) == 'torch.LongTensor' then
        data[k] = data[k]:cuda()
      else
        data[k] =  data[k]:cuda()
      end
    end
  end

  -- Fetch data using the loader
  local inputs = {{data.fc7_local, data.fc7_context, data.bbox_coordinate},{data.sentence, data.length}}
  --if opt.gpuid >= 0 then
    --nn.utils.recursiveType(inputs, 'torch.CudaTensor')
    --for k, v in pairs(data) do
    --  data[k] = v:cuda()
    --end
  --end
  local loss
  
  local fb_time = utils.timeit(function()
    local scores = protos.net:forward(inputs)
    loss = protos.crit:forward(scores, data.iou)
    -- backward path
    local dscores = protos.crit:backward(scores, data.iou)
    protos.net:backward(inputs, dscores)
  end)
  print('forward backward time:' .. fb_time)

  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- Visualization/Logging code
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- clip gradients
  -- print(string.format('claming %f%% of gradients', 100*torch.mean(torch.gt(torch.abs(grad_params), opt.grad_clip))))
  grad_params:clamp(-opt.grad_clip, opt.grad_clip)

  -----------------------------------------------------------------------------

  -- and lets get out!
  local losses = { total_loss = loss }
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
while true do  
  for n, data in trainLoader:run() do
    -- Compute loss and gradient
    local losses = lossFun(data)
    if iter % opt.losses_log_every == 0 then 
      loss_history[iter] = losses.total_loss 
    end
    print(string.format('iter %d: %s', iter, losses.total_loss))
    if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters) then
      -- evaluate the validation performance
      local val_loss, val_result = eval_split(valLoader, {val_images_use = opt.val_images_use})
      print('validation loss: ', val_loss)
      print(val_result)
      val_loss_history[iter] = val_loss
      val_results_history[iter] = val_result
      lr_history[iter] = opt.learning_rate

      local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

      -- write a (thin) json report
      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.lr_history = lr_history
      checkpoint.val_loss_history = val_loss_history
      checkpoint.val_results_history = val_results_history

      utils.write_json(checkpoint_path .. '.json', checkpoint)
      print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

      -- write the full model checkpoint as well if we did better than ever
      local current_score = -val_loss
      
      if best_val_score == nil or current_score > best_val_score then
        best_val_score = current_score
        checkpoint.best_val_score = best_val_score
        
        -- include the protos (which have weights) and save to file
        checkpoint.optim_state = optim_state
        local save_protos = {}
        save_protos.net = protos.net
        save_protos.net:clearState()
        save_protos.net:float()
        if cudnn then
          cudnn.convert(save_protos.net, nn)
        end
        checkpoint.protos = save_protos
        -- also include the vocabulary mapping so that we can use the checkpoint 
        -- alone to run on arbitrary images without the data loader
        torch.save(checkpoint_path .. '.t7', checkpoint)
        print('wrote checkpoint to ' .. checkpoint_path .. '.t7')

        -- Now go back to CUDA and cuDNN
        protos.net:cuda()
        
        if cudnn then
          cudnn.convert(protos.net, cudnn)
        end
        -- All of that nonsense causes the parameter vectors to be reallocated, so
        -- we need to reallocate the params and grad_params vectors.
        params, grad_params= protos.net:getParameters()
      end
    end

    -- decay the learning rate for both LM and CNN
    local learning_rate = opt.learning_rate

    -- perform a parameter update
    if opt.optim == 'rmsprop' then
      rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
    elseif opt.optim == 'sgd' then
      sgd(params, grad_params, opt.learning_rate)
    elseif opt.optim == 'sgdm' then
      sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
    elseif opt.optim == 'adam' then
      adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
    else
      error('bad option opt.optim')
    end

    -- stopping criterions
    iter = iter + 1
    if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
    if loss0 == nil then loss0 = losses.total_loss end
    --if losses.total_loss > loss0 * 20 then
    --  print('loss seems to be exploding, quitting.')
    --  break
    --end
    if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
  end
end