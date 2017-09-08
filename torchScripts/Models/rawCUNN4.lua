-- first layer extract 16 features , out of each 0.02 sec with 50% overlapping (0-0.02,0.01-0.03,0.02-0.04 etc)
-- we need to modify the input in correlation
require "nn"
--require "cunn"
local poolSize = 2

--local net = nn.Sequential()
local net = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization

--assume data is type cuda
--data=data:cuda()


--net:add(nn.Reshape(256+2^16))
net:add(nn.BatchNormalization(interval_size,nil,nil,false))
net:add(nn.Reshape(interval_size,1))

N = interval_size; wn = 512; stride = 256;
net:add(nn.TemporalConvolution(1,16,wn,stride))
N = math.floor((N-wn)/stride+1)

stride = 1;
net:add(nn.TemporalMaxPooling(poolSize,stride))
-----add size check---------------------------------
N = math.floor((N-poolSize)/stride+1)
net:add(nn.ReLU())

net:add(nn.Reshape(N*16))
net:add(nn.BatchNormalization(N*16,nil,nil,false))
net:add(nn.Reshape(N,16))

net:add(nn.TemporalConvolution(16,32,1))


net:add(nn.TemporalMaxPooling(poolSize))
net:add(nn.ReLU()) 
N = torch.floor(N/2)
net:add(nn.Reshape(N*32))
net:add(nn.BatchNormalization(N*32,nil,nil,false))
net:add(nn.Reshape(N,32))

net:add(nn.TemporalConvolution(32,64,1))


net:add(nn.TemporalMaxPooling(poolSize))
net:add(nn.ReLU()) 
N = torch.floor(N/2)
net:add(nn.Reshape(N*64))
net:add(nn.BatchNormalization(N*64,nil,nil,false))

net:add(nn.Dropout(0.5))

--args.nClasses = args.nClasses or 9

--args = args or {nClasses}
--args[nClasses] = args[nClasses] or 9 

feat1 = nn.Sequential():add(nn.Linear(N*64,1000))
feat1:add(nn.ReLU()) :add(nn.Dropout(0.5)):add(nn.Linear(1000,args.nClasses)):add(nn.LogSoftMax())
feat2 = nn.Sequential():add(nn.Linear(N*64,args.nClasses/3)):add(nn.LogSoftMax())

mlp = nn.ConcatTable()
mlp:add(feat1)
mlp:add(feat2)

net:add(mlp)

net:float()

net:cuda()
net:training()
-- -- -- x = torch.randn(nExample,2):float()  --:float()-> change from doubleTensor to floatTensor

-- -- -- Criterion

-- -- criterion	= nn.MSECriterion()
-- -- trainer 	= nn.StochasticGradient( mlp, criterion )
-- -- trainer.learningRate = 0.01

-- -- -- Creates nResults
-- -- trainer:train(trainX)
--MLNN params------------------------------------------------------



return net
