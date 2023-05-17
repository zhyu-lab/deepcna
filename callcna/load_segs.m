function data_seg_sep = load_segs(segFile)

fid = fopen(segFile,'r');
results = textscan(fid,'%f%f%f','HeaderLines',1);
data_seg_all = cell2mat(results);
clear results;
fclose(fid);

chromosomes = reshape(unique(data_seg_all(:,1)),1,[]);
data_seg_sep = cell(1,length(chromosomes));
for i = 1:length(chromosomes)
    tv = data_seg_all(:,1) == chromosomes(i);
    data_seg_sep{i} = data_seg_all(tv,[2 3]);
end

end