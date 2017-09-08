local net = nn.Sequential()

--assume data is type cuda
--data=data:cuda()
net:add(nn.Reshape(256*12))
net:add(nn.Linear(256*12,1000))
net:add( nn.Tanh() )
net:add(nn.Dropout(0.5))
net:add(nn.Linear(1000,nClasses))
net:add( nn.Tanh() )
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
