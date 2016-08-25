DataLoader = require 'dataloader'
opts = require 'train_opts'
opt = opts.parse(arg)
print(opt)
trainLoader, valLoader = DataLoader.create(opt)
x = trainLoader.dataset.data_pairs
trainLoader.dataset:get(1)
for n, sample in trainLoader:run() do
print(n)
if n == 100 then
break
end
end

trainLoader:restart()
for n, sample in trainLoader:run() do
print(n)
if n == 100 then
break
end
end
