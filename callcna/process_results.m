function [p_states,aCN,segments] = process_results(depend_table)
% 10/12/2022 by Zhenhua
%-----------------------------------------------------
%------overall information of the cell------
%p_states: proportions of all states
%aCN: averaged copy number
%segments: copy number segmentation results

global gamma_sep
global data_seg_sep

tv_S = depend_table(:,2)==1;
Y = depend_table(tv_S,3)'; % copy number of different entries

%initialize output parameters
segments = [];

%initialize intermediate variables
pos_dist = [];

for i = 1:length(gamma_sep) %for the ith chromosome
    post_probs = gamma_sep{i};
    data_seg = data_seg_sep{i};

    %---handle MAP states---
    %output predicted MAP states
    [temp,MAP_state] = max(post_probs,[],1);
    
    segments = [segments; ones(length(MAP_state),1)*i data_seg MAP_state'];
    pos_dist = [pos_dist; (data_seg(:,2)-data_seg(:,1)+1)];

    clear results;

end
pos_dist = pos_dist';

%---handle p_states---
p_states = zeros(length(Y),1);
for i = 1:length(Y)
    tv = segments(:,4) == i;
    if sum(tv) > 0
        p_states(i) = sum(pos_dist(tv))/sum(pos_dist);
    end
end

%---handle aCN---
aCN = Y*p_states;

end