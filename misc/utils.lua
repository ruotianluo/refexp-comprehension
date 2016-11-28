local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

-- return whether file exists or not
function utils.file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- return set of table
function utils.unique(t)
  local res = {}
  local hash = {}
  for _, v in ipairs(t) do
    if not hash[v] then
      res[#res+1] = v
      hash[v] = true
    end
  end
  return res
end

-- copy table
function utils.copyTable(t)
  local a = {}
  for i, v in ipairs(t) do a[i] = v end
  return a
end

-- shuffle table
function utils.shuffleTable(t)
  local n = #t
  while n >= 2 do
    -- n is now the last pertinent index
    local k = math.random(n) -- 1 <= k <= n
    -- Quick swap
    t[n], t[k] = t[k], t[n]
    n = n-1
  end
  return t
end

-- not self shuffle 
function utils.shuffleTable_self_taboo(t)
  local n = #t
  while n >= 2 do
    -- n is now the last pertinent index
    local k = math.random(n-1)
    -- quick swap
    t[n], t[k] = t[k], t[n] 
    n = n-1
  end
  return t
end

-- view input = torch.float(3, m, n), ranging from 0-255
function utils.viewRawImg(input)
  -- load cv
  local cv = require 'cv'
  require 'cv.imgcodecs'
  require 'cv.highgui'
  -- convert to opencv format
  local height, width = input:size(2), input:size(3)
  local im = input:byte()
  -- change order and transpose
  im = im:index(1, torch.LongTensor{3,2,1})
  im = im:transpose(1,2):transpose(2,3)
  -- opencv's load
  local cv_im = torch.zeros(height, width, 3):byte()
  cv_im[{ {1, height}, {1, width}, {} }] = im
  cv.imshow {'im', cv_im}
  cv.waitKey {delay=0}
end
-- compute IoU of two boxes
function utils.IoU(box1, box2)

  local x1, y1, w1, h1 = unpack(box1)
  local x2, y2, w2, h2 = unpack(box2)

  local inter_x1 = math.max(x1, x2)
  local inter_y1 = math.max(y1, y2)
  local inter_x2 = math.min(x1+w1-1, x2+w2-1)
  local inter_y2 = math.min(y1+h1-1, y2+h2-1)
  local inter
  if inter_x1 < inter_x2 and inter_y1 < inter_y2 then
    inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
  else
    inter = 0
  end
  union = w1*h1+w2*h2-inter
  return inter/union
end

-- Count the time of a function
function utils.timeit(f)
  utils.timer = utils.timer or torch.Timer()
  cutorch.synchronize()
  utils.timer:reset()
  f()
  cutorch.synchronize()
  return utils.timer:time().real
end

return utils
