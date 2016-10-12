require 'nn'
local utils = require 'misc.utils'

net_utils = {}

--[[
scale by multiplying a learnable multiplie, that is initialized by init_norm.
]]
function net_utils.scale_layer(d, init_norm)
	local m = nn.CMul(d)
	m.weight:fill(init_norm)
	return m
end

function net_utils.build_visnet_lc(opt)
	local dropout = utils.getopt(opt, 'drop_prob_vis', 0.5)
	local init_norm = utils.getopt(opt, 'init_norm', 20)
	local use_ann = utils.getopt(opt, 'use_ann', 1)
	local use_context = utils.getopt(opt, 'use_context', 1)
	local use_location = utils.getopt(opt, 'use_location', 1)
	-- construct parallel layer
	local M = nn.ConcatTable()
	local dim = 0
	-- region cnn
	if use_ann > 0 then
		local ann_view = nn.Sequential()
		ann_view:add(nn.SelectTable(1))
		ann_view:add(nn.Linear(4096, opt.image_l1_size))
				:add(nn.Normalize(2))
				:add(net_utils.scale_layer(opt.image_l1_size, init_norm))
		M:add(ann_view)
		dim = dim + opt.image_l1_size
	end
	-- global cnn
	if use_context > 0 then
		local cxt_view = nn.Sequential()
		cxt_view:add(nn.SelectTable(2))
		cxt_view:add(nn.Linear(4096, opt.image_l1_size))
				:add(nn.Normalize(2))
				:add(net_utils.scale_layer(opt.image_l1_size, init_norm))
      	        :add(nn.Replicate(101))
                :add(nn.Squeeze(2))
		M:add(cxt_view)
		dim = dim + opt.image_l1_size
	end
	-- location 
	if use_location > 0 then
		local location_view = nn.Sequential()
		location_view:add(nn.SelectTable(3))
		location_view:add(nn.Normalize(2))
					 :add(net_utils.scale_layer(8, init_norm))
		M:add(location_view)
		dim = dim + 8
	end
	-- jointly embed them
	local jemb_part = nn.Sequential()
	jemb_part:add(M)
			 :add(nn.JoinTable(2))
			 :add(nn.Linear(dim, opt.image_l2_size))
			 :add(nn.Dropout(dropout))
	return jemb_part
end

function net_utils.build_visnet_rt(opt)
	local dropout = utils.getopt(opt, 'drop_prob_vis', 0.5)
	local init_norm = utils.getopt(opt, 'init_norm', 20)
	local use_ann = utils.getopt(opt, 'use_ann', 1)
	local use_context = utils.getopt(opt, 'use_context', 1)
	local use_location = utils.getopt(opt, 'use_location', 1)
	-- construct parallel layer
	local M = nn.ConcatTable()
	local dim = 0
	-- region cnn
	if use_ann > 0 then
		local ann_view = nn.Sequential()
		ann_view:add(nn.SelectTable(1))
		ann_view:add(nn.Linear(4096, opt.image_l1_size))
				:add(nn.ReLU())
    			--:add(nn.BatchNormalization(1000))
    			:add(nn.Dropout(dropout))
		M:add(ann_view)
		dim = dim + opt.image_l1_size
	end
	-- global cnn
	if use_context > 0 then
		local cxt_view = nn.Sequential()
		cxt_view:add(nn.SelectTable(2))
		cxt_view:add(nn.Linear(4096, opt.image_l1_size))
				:add(nn.ReLU())
    			--:add(nn.BatchNormalization(1000))
      	        :add(nn.Replicate(101))
                :add(nn.Squeeze(2))
    			:add(nn.Dropout(dropout))
		M:add(cxt_view)
		dim = dim + opt.image_l1_size
	end
	-- location 
	if use_location > 0 then
		local location_view = nn.Sequential()
		location_view:add(nn.SelectTable(3))
		location_view:add(nn.Identity())
			--:add(nn.BatchNormalization(5))
			--:add(nn.Dropout(dropout))
		M:add(location_view)
		dim = dim + 8
	end
	-- jointly embed them
	local jemb_part = nn.Sequential()
	jemb_part:add(M)
			 :add(nn.JoinTable(1,1))
  			 :add(nn.Linear(dim,opt.image_l2_size))
  			 :add(nn.ReLU())
  			 -- :add(nn.BatchNormalization(opt.image_l2_size))
  			 :add(nn.Dropout(dropout))
  
	return jemb_part
end

function net_utils.build_visnet_old(opt)
  local vis_net = nn.Sequential():add(nn.ParallelTable())
  -- reg
  vis_net:get(1):add(nn.Sequential()
    :add(nn.Linear(4096, opt.image_l1_size))
    :add(nn.ReLU()))
    --:add(nn.BatchNormalization(self.image_l1_size)))
  -- context
  vis_net:get(1):add(
    nn.Sequential():add(nn.Linear(4096, opt.image_l1_size))
      :add(nn.ReLU())
      --:add(nn.BatchNormalization(self.image_l1_size))
      :add(nn.Replicate(101))
      :add(nn.Squeeze(2)))

  -- locaition
  vis_net:get(1):add(nn.BatchNormalization(8))

  vis_net:add(nn.JoinTable(1,1))
  vis_net
    :add(nn.Linear(opt.image_l1_size*2+8,opt.image_l2_size))
    :add(nn.ReLU())
    :add(nn.BatchNormalization(opt.image_l2_size))

  return vis_net

end

return net_utils