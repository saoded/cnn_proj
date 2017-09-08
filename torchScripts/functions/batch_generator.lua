function batch_generator(mixed_xtrain_inds,iBatch)

batchXi = torch.Tensor(args.batch_size,interval_size)--,1)
batchXi = batchXi:cuda()
batchYi = torch.Tensor(args.batch_size)
batchYi = batchYi:cuda()
batchYi2 = torch.Tensor(args.batch_size)
batchYi2 = batchYi2:cuda()
-- iBatch*(args.batch_size-1)+1;
i = 1;
iTrain = (iBatch-1)*args.batch_size+1

repeat
	for iClass =1,args.nClasses do
		j = torch.ceil(i/args.nClasses)
		--print(nIntervals)
		--print(mixed_xtrain_inds[iClass]:size(1)/nIntervals)
		--print(j+(iBatch-1)*args.batch_size)
		temp = j+(iBatch-1)*(args.batch_size/args.nClasses)
		local iSong 	= torch.ceil(mixed_xtrain_inds[iClass][temp]/nIntervals)		
		local iInterval = mixed_xtrain_inds[iClass][temp]-(iSong-1)*nIntervals	
		--print(iInterval)
		--print(iSong)
		local startInd 	= train.intervals_start[iSong][iInterval]
		--print(startInd)
		--print(iSong)
		local endInd 	= startInd+interval_size-1
		local xi	= xtot[iClass]:select(1,iSong):sub(startInd,endInd)--:reshape(interval_size,1)
		batchXi[i] = xi
		batchYi[i] = iClass
		batchYi2[i] = torch.ceil(iClass/3)
		iTrain = iTrain+1;
		i = i+1
		--print(i)
		--print(iTrain)
		if ((i>args.batch_size) or (iTrain > train.size*args.nClasses)) then
			break;
		end
	end
	
until  i>args.batch_size or iTrain > train.size*args.nClasses

if i <  args.batch_size then --end of the train set, incase batch isnt full
	batchXi = batchXi:sub(1,i-1)
 	batchYi = batchYi:sub(1,i-1)
	batchYi2 = batchYi2:sub(1,i-1)
end
--print(batchXi:size())
return batchXi,batchYi,batchYi2

end
