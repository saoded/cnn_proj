poolSize=2
local net = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization

--assume data is type cuda

--data=data:cuda()

nFrames = 256
net:add(nn.TemporalConvolution(13,32,1))
--net:add(nn.TemporalConvolution(outputFrameSize[1], outputFrameSize[2], 1))
net:add(nn.ReLU()) 
net:add(nn.TemporalMaxPooling(poolSize))
nFrames = nFrames/2
net:add(nn.TemporalConvolution(32,64,1))
--net:add(nn.TemporalConvolution(outputFrameSize[2], outputFrameSize[3], 1))
net:add(nn.ReLU()) 
net:add(nn.TemporalMaxPooling(poolSize))
nFrames = nFrames/2

net:add(nn.Reshape(nFrames*64))
net:add(nn.Linear(nFrames*64,nClasses))
net:add(nn.LogSoftMax())

net:float()

--net:cuda()

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
