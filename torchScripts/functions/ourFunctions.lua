local ourFunctions = {}
--train 1 sample---------------------------------------------------
function ourFunctions.f_eval()
	grad_W:zero()
	local y = net:forward(batchXi)
	local j=1
	m = batchYi:size(1)

	repeat
		train_confusion:add(y[1][j],batchYi[j])
		train_confusion2:add(y[2][j],batchYi2[j])
		j= j+1
	until j>m
        local dE_dy = loss:backward(y,{batchYi,batchYi2})
	--local dE_dy = loss:backward(y[2],batchYi2)
        net:backward(batchXi, dE_dy)
        return 0, grad_W
end
--create test confusion -------------------------------------------
function ourFunctions.test()
	net:evaluate()
	local j	= 1
	iBatch = 1
	for iClass =1,args.nClasses do
		local yi = iClass;
		for jSong = 1,test.nSongs do
			local iSong = train.nSongs+jSong
			res= torch.FloatTensor(nIntervals,args.nClasses);
			res2= torch.FloatTensor(nIntervals,args.nClasses/3);
			res3= torch.FloatTensor(nIntervals,3);
			for iInterval =1,nIntervals do
				local startInd = test.intervals_start[jSong][iInterval]
				local endInd 	= startInd+interval_size-1
				local xi	= xtot[iClass]:select(1,iSong):sub(startInd,endInd):reshape(1,interval_size):cuda()
				local testYRes = net:forward(xi)
				--print(torch.exp(testYRes):squeeze())

				res[iInterval] = torch.exp(testYRes[1]):squeeze():float()
				res2[iInterval] = torch.exp(testYRes[2]):squeeze():float()
				lol = (torch.ceil(iClass/3))
				test_confusion:add(testYRes[1][1], iClass)
				test_confusion2:add(testYRes[2][1], lol)
				if (iClass > 3 and iClass < 13) then
					--test_confusion9artists:add(testYRes[1][1]:index(1,{4,5,6,7,8,9,10,11,12}), iClass)
					--print(testYRes[2][1])

					indices = torch.LongTensor({2,3,4})
					res3genres = testYRes[2][1]:index(1,indices)-1;
					res3[iInterval] = torch.exp(res3genres):squeeze():float()
					test_confusion3genres:add(testYRes[2][1]:index(1,indices)-1, lol-1)
				end
			end

			testYResAVG = res:mean(1):squeeze()
			testYResAVG2 = res2:mean(1):squeeze()
			testYResAVG3genres = res3:mean(1):squeeze()
			test_confusionAVG:add(testYResAVG:cuda(),yi)
			test_confusionAVG2:add(testYResAVG2:cuda(),torch.ceil(iClass/3))
			if (iClass > 3 and iClass < 13) then
				test_confusionAVG3genres:add(testYResAVG3genres,torch.ceil(iClass/3)-1)
			end
		end
	end
end

-- generates x_start_inds : x_start_inds[i][j] is the start ind of the j interval in song i.
-- note: all classes are given the same start index
function ourFunctions.generate_intervals_start(song_size,nIntervals,nSongs,interval_size )

	local x_start_inds =  torch.DoubleTensor(nSongs,nIntervals)
	x_start_inds:apply(function()
	local randIndStart = torch.randperm(torch.floor((song_size-interval_size)/5000))
	return (1+(randIndStart[1])*5000) end)
	return x_start_inds:int()
end

function ourFunctions.mix_x_inds(nClasses, n_inputs)
	mixed_x_inds={}
	for iClass =1,args.nClasses do
		mixed_x_inds[iClass] = torch.randperm(n_inputs)
	end
	return mixed_x_inds

end

return ourFunctions
