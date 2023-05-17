clc;
clear;
fclose all;

ds = 'ploidy_4_tree_2_clones_4_cells_100';

callcna(['./data/' ds '/lrc.txt'],['./data/' ds '/seg.txt'],['./results/' ds],10);
plot_results(['./data/' ds '/lrc.txt'],['./results/' ds]);
