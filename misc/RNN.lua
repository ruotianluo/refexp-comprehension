require 'nn'
require 'rnn'

local BiDynamicRNN, parent = torch.class('BiDynamicRNN', 'nn.Module')

function BiDynamicRNN:__init(cell_fw, cell_bw, time_major)
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
  self.time_major = time_major or false

  local fwd = nn.Sequential()
    --:add(nn.SplitTable(1))
    :add(cell_fw:maskZero(1))

  -- internally, rnn will be wrapped into a Recursor to make it an AbstractRecurrent instance.
  fwdSeq = nn.Sequencer(fwd)

  -- backward rnn (will be applied in reverse order of input sequence)
  local bwd = nn.Sequential()
    --:add(nn.SplitTable(1))
    :add(cell_bw:maskZero(1))
  bwdSeq = nn.Sequencer(bwd)

  -- merges the output of one time-step of fwd and bwd rnns.
  -- You could also try nn.AddTable(), nn.Identity(), etc.
  local merge = nn.JoinTable(1, 1)
  mergeSeq = nn.Sequencer(merge)

  -- Assume that two input sequences are given (original and reverse, both are right-padded).
  -- Instead of ConcatTable, we use ParallelTable here.
  local parallel = nn.ParallelTable()
  parallel:add(fwdSeq):add(bwdSeq)
  local brnn = nn.Sequential()
    :add(parallel)
    --:add(nn.ZipTable())
    --:add(mergeSeq)
    --:add(nn.Sequencer(nn.Unsqueeze(1)))
    --:add(nn.JoinTable(1))
    :add(nn.JoinTable(3))

  self.brnn = brnn
end

function BiDynamicRNN:training()
  parent.training(self)
  self.brnn:training()
end

function BiDynamicRNN:evaluate()
  parent.evaluate(self)
  self.brnn:evaluate()
end

function BiDynamicRNN:clearState()
  self.brnn:clearState()
end

function BiDynamicRNN:parameters()
  return self.brnn:parameters()
end

function BiDynamicRNN:updateOutput(inputs)
--[[
inputs: a table; inputs[1] is batch_size * max_length * size or max_length * batch_size * size
inputs[2] is sequence length, including the length of each sequence
]]--
  local sequence_length = inputs[2]
  local max_length = torch.max(sequence_length)

  if self.time_major then
    self.rnn_inputs = inputs[1][{{1, max_length}}]
  else
    self.rnn_inputs = inputs[1][{{},{1, max_length}}]:transpose(1, 2):contiguous()
  end

  local batch_size = self.rnn_inputs:size(2)
  -- Generate the inversenet
  self.rnn_inputs_rev = self.rnn_inputs.new(#self.rnn_inputs):zero()
  for i = 1, batch_size do
    for k = 1, sequence_length[i] do
      self.rnn_inputs_rev[k][i]:copy(self.rnn_inputs[sequence_length[i] - k + 1][i])
    end
  end
  local outputs = self.brnn:forward({self.rnn_inputs, self.rnn_inputs_rev})
  local output_states = self.rnn_inputs.new(outputs[1]:size())
  for i = 1, batch_size do
    output_states[i]:copy(outputs[sequence_length[i]][i])
  end
  self.output = output_states
  return self.output
end

function BiDynamicRNN:updateGradInput(inputs, gradOutput)
  local sequence_length = inputs[2]
  local batch_size = self.rnn_inputs:size(2)

  self.gradInput = {inputs[1].new(#inputs[1]):zero(),{}}

  local max_length = torch.max(sequence_length)

  local output_feat_size = gradOutput:size(2)
  local brnn_grad_output = gradOutput.new(max_length, batch_size, output_feat_size):zero()
  for i = 1, batch_size do
    brnn_grad_output[sequence_length[i]][i]:copy(gradOutput[i])
  end
  local brnn_grad_input = self.brnn:backward({self.rnn_inputs, self.rnn_inputs_rev}, brnn_grad_output)

  for i = 1, batch_size do
    for k = 1, sequence_length[i] do
      brnn_grad_input[1][k][i]:add(brnn_grad_input[2][sequence_length[i] - k + 1][i])
    end
  end

  brnn_grad_input = brnn_grad_input[1]

  if self.time_major then
    self.gradInput[1][{{1, max_length}}]:copy(brnn_grad_input)
  else
    self.gradInput[1][{{},{1, max_length}}]:copy(brnn_grad_input:transpose(1, 2))
  end

  return self.gradInput
end

function BiDynamicRNN:clearState()
  self.rnn_inputs = nil
  self.rnn_inputs_rev = nil
  self.gradInput = nil
  self.brnn:clearState()
  return parent.clearState(self)
end

