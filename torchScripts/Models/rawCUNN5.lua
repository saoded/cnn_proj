require "nn"
require "cunn"
local poolSize = 2
interval_size =interval_size or 10*22050;

--local net = nn.Sequential()
net = nn.Sequential()

net:add(nn.BatchNormalization(interval_size,nil,nil,false))
net:add(nn.Reshape(interval_size,1))

---conv 1
N = interval_size; wn = 512; stride = 256;
net:add(nn.TemporalConvolution(1,16,wn,stride))
N = math.floor((N-wn)/stride+1)

stride = 1;
net:add(nn.TemporalMaxPooling(poolSize,stride))
N = (N-poolSize)/stride+1
net:add(nn.ReLU())

net:add(nn.Reshape(N*16))
net:add(nn.BatchNormalization(N*16,nil,nil,false))
net:add(nn.Reshape(N,16))


---conv 2
wn =8;  stride = wn/2;
net:add(nn.TemporalConvolution(16,32,wn,stride))
N = math.floor((N-wn)/stride+1)

net:add(nn.TemporalMaxPooling(poolSize))
net:add(nn.ReLU()) 
N = torch.floor(N/2)
net:add(nn.Reshape(N*32))
net:add(nn.BatchNormalization(N*32,nil,nil,false))
net:add(nn.Reshape(N,32))

---conv 3

net:add(nn.TemporalConvolution(32,64,wn,stride))
N = math.floor((N-wn)/stride+1)
net:add(nn.TemporalMaxPooling(poolSize))
net:add(nn.ReLU()) 
N = torch.floor(N/2)
net:add(nn.Reshape(N*64))
net:add(nn.BatchNormalization(N*64,nil,nil,false))
net:add(nn.Reshape(N,64))


---conv 4
net:add(nn.TemporalConvolution(64,128,wn,stride))
N = math.floor((N-wn)/stride+1)
net:add(nn.TemporalMaxPooling(poolSize))
net:add(nn.ReLU()) 
N = torch.floor(N/2)
net:add(nn.Reshape(N*128))
net:add(nn.BatchNormalization(N*128,nil,nil,false))
--net:add(nn.Reshape(N,128))

feat1 = nn.Sequential():add(nn.Linear(N*128,args.nClasses)):add(nn.LogSoftMax())
feat2 = nn.Sequential():add(nn.Linear(N*128,args.nClasses/3)):add(nn.LogSoftMax())

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
