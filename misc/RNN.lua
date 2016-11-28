require 'nn'
require 'rnn'
require 'torchx'
local utils = require 'misc.utils'

local DynamicRNN, parent = torch.class('nn.DynamicRNN', 'nn.Module')

function DynamicRNN:__init(cell, opt)
--[[
"""Creates a dynamic version of bidirectional recurrent neural network.
  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs. The input_size of forward and
  backward cell must match. The initial state for both directions is zero by
  default (but can be set optionally) and no intermediate states are ever
  returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not
  given.
  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, input_size]`.
      If time_major == True, this must be a tensor of shape:
        `[max_time, batch_size, input_size]`.
      [batch_size, input_size].
    sequence_length: An int32/int64 vector, size `[batch_size]`,
      containing the actual lengths for each of the sequences.
  Returns:
    A tuple (outputs, output_states) where:
      outputs: A tuple (output_fw, output_bw) containing the forward and
        the backward rnn output `Tensor`.
        If time_major == False (default),
          output_fw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[batch_size, max_time, cell_bw.output_size]`.
        If time_major == True,
          output_fw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_fw.output_size]`
          and output_bw will be a `Tensor` shaped:
          `[max_time, batch_size, cell_bw.output_size]`.
        It returns a tuple instead of a single concatenated `Tensor`, unlike
        in the `bidirectional_rnn`. If the concatenated one is preferred,
        the forward and backward outputs can be concatenated as
        `tf.concat(2, outputs)`.
      output_states: A tuple (output_state_fw, output_state_bw) containing
        the forward and the backward final states of bidirectional rnn.
  Raises:
    TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
  """
]]--
  -- forward rnn
  self.dropout = utils.getopt(opt, 'dropout', 0.2)
  self.time_major = utils.getopt(opt, 'time_major', 0)
  self.state_type = utils.getopt(opt, 'state_type', 'avg')
  self.layer_num = utils.getopt(opt, 'layer_num', 1)
  self.input_size = utils.getopt(opt, 'input_size', 512)
  self.hidden_size = utils.getopt(opt, 'hidden_size', 512)
  self.bidirectional = utils.getopt(opt, 'bidirectional', 1)
  -- self.output_size = utils.getopt(opt, 'output_size', 512)

  local rnn = nn.Sequential()

  rnn:add(nn.SplitTable(1))

  for i = 1, self.layer_num do
    if self.bidirectional == 1 then
      if i == 1 then
        rnn:add(nn.BiSequencer(cell(self.input_size, self.hidden_size):remember('both'):maskZero(1)))
      else
        rnn:add(nn.BiSequencer(cell(self.hidden_size, self.hidden_size):remember('both'):maskZero(1)))
      end
      if i < self.layer_num then
        local nonlinear = nn.Sequential()
        nonlinear:add(nn.Linear(self.hidden_size * 2, self.hidden_size))
        nonlinear:add(nn.ReLU())
        -- nonlinear:add(nn.Sigmoid())
        rnn:add(nn.Sequencer(nn.MaskZero(nonlinear, 1)))
      end
    else
      rnn:add(nn.Sequencer(cell(self.input_size, self.hidden_size):remember('both'):maskZero(1)))
    end
    if i < self.layer_num then
      rnn:add(nn.Sequencer(nn.Dropout(self.dropout)))
    end
  end
  rnn:add(nn.Sequencer(nn.Unsqueeze(1)))
  rnn:add(nn.JoinTable(1))
  self.rnn = rnn
  
  self.module = self.rnn
  self.modules = {self.rnn}
end

function DynamicRNN:training()
  parent.training(self)
  self.rnn:training()
  for k,v in pairs(self.rnn:findModules('nn.Dropout')) do
    v.p = self.dropout
  end
end

function DynamicRNN:evaluate()
  parent.evaluate(self)
  self.rnn:evaluate()
  for k,v in pairs(self.rnn:findModules('nn.Sequencer')) do
    v:training()
  end
  for k,v in pairs(self.rnn:findModules('nn.Dropout')) do
    v.p = 0
  end
end

function DynamicRNN:clearState()
  self.rnn:clearState()
end

function DynamicRNN:parameters()
  return self.rnn:parameters()
end

function DynamicRNN:getSeqLength(input)
  local seq_len
  local feat_norm
  if self.time_major == 1 then
    seq_len = torch.LongTensor(input:size(2))
    feat_norm = torch.norm(input, 2, 3):squeeze(3):t() -- make it batch major
  else
    seq_len = torch.LongTensor(input:size(1))
    feat_norm = torch.norm(input, 2, 3):squeeze(3)
  end
  for i = 1, seq_len:size(1) do
    seq_len[i] = torch.find(feat_norm[i], 0)[1] or (feat_norm:size(2) + 1)
  end
  seq_len:add(-1)
  return seq_len
end

function DynamicRNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  if torch.type(inputs) ~= 'table' then
    inputs = {inputs}
    inputs[2] = self:getSeqLength(inputs[1])
  end
  local sequence_length = inputs[2]
  self.sequence_length = sequence_length
  local max_length = torch.max(sequence_length)

  if self.time_major == 1 then
    self.rnn_inputs = inputs[1][{{1, max_length}}]
  else
    self.rnn_inputs = inputs[1][{{},{1, max_length}}]:transpose(1, 2):contiguous()
  end

  local batch_size = self.rnn_inputs:size(2)

  -- Ensure the feature out of the sequence length are zero
  for i = 1, batch_size do
    if sequence_length[i]+1 < max_length then
      self.rnn_inputs[{{sequence_length[i]+1, max_length}, i}]:zero()
    end
  end

  local outputs = self.rnn:forward(self.rnn_inputs)
  local output_states
  if self.state_type == 'final' then
    output_states = self.rnn_inputs.new(outputs[1]:size())
    if self.bidirectional == 1 then
      for i = 1, batch_size do
        output_states[{i, {1, outputs:size(3)/2}}]:copy(outputs[{sequence_length[i], i, {1, outputs:size(3)/2}}])
      end
      output_states[{{}, {outputs:size(3)/2 + 1, -1}}]:copy(outputs[{1, {}, {outputs:size(3)/2 + 1, -1}}])
    else
      for i = 1, batch_size do
        output_states[i]:copy(outputs[sequence_length[i]][i])
      end
    end
  else
    output_states = torch.sum(outputs, 1):squeeze(1)
    output_states:cdiv(sequence_length:typeAs(output_states):view(-1, 1):expandAs(output_states))
  end
  self.output = output_states
  return self.output
end

function DynamicRNN:updateGradInput(inputs, gradOutput)
  local single_input = false -- Whether the inputs is a tensor or a table
  if torch.type(inputs) ~= 'table' then
    inputs = {inputs}
    inputs[2] = self.sequence_length
    single_input = true
  end
  local sequence_length = self.sequence_length
  local batch_size = self.rnn_inputs:size(2)

  self.gradInput = {inputs[1].new(#inputs[1]):zero(), {}}

  local max_length = torch.max(sequence_length)

  local output_feat_size = gradOutput:size(2)
  local rnn_grad_output
  if self.state_type == 'final' then
    rnn_grad_output = gradOutput.new(max_length, batch_size, output_feat_size):zero()
    for i = 1, batch_size do
      rnn_grad_output[sequence_length[i]][i][{{1, output_feat_size / 2}}]:copy(gradOutput[i][{{1, output_feat_size / 2}}])
    end
    rnn_grad_output[1][{{}, {output_feat_size / 2 + 1, -1}}]:copy(gradOutput[{{}, {output_feat_size / 2 + 1, -1}}])
  else
    rnn_grad_output = torch.cdiv(gradOutput, sequence_length:typeAs(gradOutput):view(-1, 1):expandAs(gradOutput))
    rnn_grad_output = rnn_grad_output:view(1, batch_size, output_feat_size):expand(max_length, batch_size, output_feat_size)
  end
  local rnn_grad_input = self.rnn:backward(self.rnn_inputs, rnn_grad_output)

  if self.time_major == 1 then
    self.gradInput[1][{{1, max_length}}]:copy(rnn_grad_input)
  else
    self.gradInput[1][{{},{1, max_length}}]:copy(rnn_grad_input:transpose(1, 2))
  end

  if single_input then
    self.gradInput = self.gradInput[1]
  end

  return self.gradInput
end

function DynamicRNN:clearState()
  self.rnn_inputs = nil
  self.rnn_inputs_rev = nil
  self.gradInput = nil
  self.sequence_length = nil
  self.rnn:clearState()
  return parent.clearState(self)
end

