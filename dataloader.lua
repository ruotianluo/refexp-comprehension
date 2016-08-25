--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.create(opt, splits)
  -- The train and val loader
  local loaders = {}

  for i, split in ipairs(splits) do
    local dataset = datasets.create(opt, split)
    loaders[i] = M.DataLoader(dataset, opt, split)
  end

  return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
  local manualSeed = opt.seed
  local function init()
    require('datasets/' .. opt.dataset)
  end
  local function main(idx)
    if manualSeed ~= 0 then
      torch.manualSeed(manualSeed + idx)
    end
    torch.setnumthreads(1)
    _G.dataset = dataset
    return dataset:size()
  end

  self.dataset = dataset

  local threads, sizes = Threads(opt.nThreads, init, main)
  self.threads = threads
  self.__size = sizes[1][1]
  self.batchSize = 1
end

function DataLoader:size()
  return math.ceil(self.__size / self.batchSize)
end

function DataLoader:run()
  local threads = self.threads
  local size, batchSize = self.__size, self.batchSize
  local perm = torch.randperm(size)

  local idx, sample = 1, nil
  local function enqueue()
    while idx <= size and threads:acceptsjob() do
      local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
      threads:addjob(
        function(indices)
          local sample
          for i, idx in ipairs(indices:totable()) do
            sample = _G.dataset:get(idx)
          end
          collectgarbage()
          return sample
        end,
        function(_sample_)
          sample = _sample_
        end,
        indices
      )
      idx = idx + batchSize
    end
  end

  local n = 0
  local function loop()
    enqueue()
    if not threads:hasjob() then
      return nil
    end
    threads:dojob()
    if threads:haserror() then
      threads:synchronize()
    end
    enqueue()
    n = n + 1
    return n, sample
  end

  return loop
end

function DataLoader:resetThreads()
  self.threads:synchronize()
end

return M.DataLoader