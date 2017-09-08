-- first layer extract 16 features , out of each 0.02 sec with 50% overlapping (0-0.02,0.01-0.03,0.02-0.04 etc)
-- we need to modify the input in correlation
local poolSize = 2

local net = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization

--assume data is type cuda
--data=data:cuda()
net:add(nn.Reshape(256+2^16))
net:add(nn.BatchNormalization(256+2^16,nil,nil,false))
net:add(nn.Reshape(256+2^16,1))
net:add(nn.TemporalConvolution(1,16,512,256))
nFrames = 256
net:add(nn.ReLU())



net:add(nn.TemporalMaxPooling(poolSize))
nFrames = nFrames/2
--net:add(nn.View(545554))

net:add(nn.Reshape(nFrames*16))
net:add(nn.BatchNormalization(nFrames*16,nil,nil,false))
net:add(nn.Reshape(nFrames,16))

---------------------new layer--------------------------------
net:add(nn.TemporalConvolution(16,16,20))

nFrames = 109
net:add(nn.Reshape(nFrames*16))

net:add(nn.BatchNormalization(nFrames*16,nil,nil,false))
net:add(nn.Reshape(nFrames,16))
--------------------------------------------------------------

net:add(nn.TemporalConvolution(16,64,1))

net:add(nn.ReLU()) 
net:add(nn.TemporalMaxPooling(poolSize))
nFrames = torch.floor(nFrames/2)
net:add(nn.Reshape(nFrames*64))
net:add(nn.BatchNormalization(nFrames*64,nil,nil,false))
net:add(nn.Reshape(nFrames,64))

net:add(nn.TemporalConvolution(64,128,1))

net:add(nn.ReLU()) 
net:add(nn.TemporalMaxPooling(poolSize))
nFrames = nFrames/2

net:add(nn.Reshape(nFrames*128))
net:add(nn.BatchNormalization(nFrames*128,nil,nil,false))

net:add(nn.Dropout(0.5))
net:add(nn.Linear(nFrames*128,nClasses))
net:add(nn.LogSoftMax())

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
