require 'misc.SimNet'
require 'nn'
tnt = require 'torchnet'
opt = {}
opt.rnn_size = 512
opt.image_l1_size = 2048
opt.image_l2_size = 1024
opt.vocab_size = 10
opt.score_type = 'dot'
opt.visnet_type = 'rt'
opt.lang_embed_type = 'rnn'

sentence = torch.LongTensor({{11,1,2,3,4,5,6,7,12}})
length = torch.LongTensor({9})

inputs = {{torch.randn(101,4096), torch.randn(1, 4096), torch.randn(101,8)}, {sentence, length}}

net = SimNet(opt)

--require 'cutorch'
--require 'cunn'
nn.utils.recursiveType(inputs[1], 'torch.FloatTensor')
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

print(inputs[1][1]:type())
net:float()

scores = net:forward(inputs)
net:backward(inputs, scores)

net:evaluate()
print(net:findModules('BiDynamicRNN')[1].train)
