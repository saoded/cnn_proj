local tbl = {}
local arg_name = 'class'

-- artists class
for i=1,args.nClasses do 
	local curr_arg_name = arg_name..i
	tbl[i] = args[curr_arg_name]
end

artists_numbers = torch.IntTensor(tbl)
artists_names = {}
file = torch.DiskFile('bash/class_list','r')
txt = file:readString('*a')
txt = txt:split('Artists:')
txt = txt[2]
for i=1,args.nClasses do 
	temp = txt:split(artists_numbers[i]..' ')
	next_artist_number = artists_numbers[i]+1
	temp = temp[2]:split(next_artist_number..' ')
	artists_names[i] = temp[1]
end 
genres_names1 = {}
genres_names1[1] = 'Alternative Rock'
genres_names1[2] = 'Punk'
genres_names1[3] = 'Hip Hop'
genres_names1[4] = 'R&B'
genres_names1[5] = 'Heavy Metal'
genres_numbers = {}
genres_numbers = torch.IntTensor({2,14,10,15,9}) 
nGenres = args.nClasses/3
genres_names = {}
for i=1,nGenres do
	genres_names[i] = genres_names1[i]
end	
return {artists_names,artists_numbers,genres_names,genres_numbers}

