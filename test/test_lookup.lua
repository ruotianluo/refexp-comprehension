require 'misc.SimNet'
require 'nn'
require 'rnn'
tnt = require 'torchnet'
opt = {}
opt.rnn_size = 512
opt.image_l1_size = 2048
opt.image_l2_size = 1024
opt.vocab_size = 10
opt.similarity = 'dot'

sentence = torch.LongTensor({{11,1,2,3,4,5,6,7,12}})
length = torch.LongTensor({9})

inputs = sentence

net = nn.Sequencer(nn.LookupTableMaskZero(opt.vocab_size + 2, opt.rnn_size)

--require 'cutorch'
--require 'cunn'
nn.utils.recursiveType(inputs, 'torch.FloatTensor')
--nn.utils.recursiveType(inputs[2][1], 'torch.FloatTensor')
--tnt.utils.table.foreach(inputs, 
--	function(v)
--		v:zero()
--		return 0
		--if torch.type(v) ~= 'torch.LongTensor' then 
		--	return v:float()
		--end
--	end, 
--	true)

net:float()

scores = net:forward(inputs)
net:backward(inputs, scores)
print(123)
