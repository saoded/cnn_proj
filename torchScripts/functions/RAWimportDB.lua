local matio = require "matio"
require "audio"


--global vars
--nClasses
--desiredClasses
--desired input mfcc/raw
--Import raw dataset---------------------------------------------------
--------Manually Change------------------------------------------------
mat_folder = '../DB/'
---------------------------------------------------------------
xtot_num={}
xtot ={}
input_train_inds ={}
for i=1,args.nClasses do
	xtot_num[i] =i;
	xtot[i] = i;
	input_train_inds[i]=i
end

local train_test_buffer = torch.IntTensor(args.nClasses) 
local test_sizeV = torch.IntTensor(args.nClasses) 
local train_sizeV = torch.IntTensor(args.nClasses) 
local class_size = torch.IntTensor(args.nClasses)	;collectgarbage()
local class_size_used = torch.IntTensor(args.nClasses)

local temp = matio.load(mat_folder..'tags.mat')
local mat_ytot = temp.tags.artists:float()
local mat_genres = temp.tags.genres

mat_ytot = mat_ytot:select(2,1)
local nData 		= mat_ytot:size(1)
local mat_xtot 		= torch.range(1,nData)

temp = audio.load(mat_folder..'raw_wav/1.wav')
local song_size		= temp:size(2)--1323000


for i=1,args.nClasses do
	
	local ytot_inds = torch.eq(mat_ytot,artists_numbers[i])
	
	class_size[i] = ytot_inds:sum()
	if(class_size[i]<args.songsPerClass) then
		error('class '..i..' is too small, change it or lower input size')
	end
	xtot_num[i] = mat_xtot[ytot_inds]
	if (mat_genres[xtot_num[i][1]][1] ~= genres_numbers[torch.ceil(i/3)]) then
		error('not the correct genre: '..i)
	end
		
end

collectgarbage()

train = {}
test = {}
train.nSongs = torch.floor(0.9*args.songsPerClass)
test.nSongs = args.songsPerClass-train.nSongs
song_numV =torch.Tensor(args.songsPerClass*args.nClasses)

for iClass =1,args.nClasses do

	inds  = torch.linspace(1,class_size[iClass],args.songsPerClass)
	xtot[iClass] = torch.FloatTensor(args.songsPerClass,song_size)	
	for iSong=1,args.songsPerClass do
		song_num = xtot_num[iClass][torch.floor(inds[iSong])]
		song_numV[iSong+args.songsPerClass*(iClass-1)] = song_num
		xtot[iClass][iSong] = audio.load(mat_folder..'raw_wav/'..song_num..'.wav'):resize(song_size)
	end
	collectgarbage()
end


return {xtot,train,test}
