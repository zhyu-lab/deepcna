function [LL_all,paras,p_states,aCN,segments] = screening(init_paras,depend_table,thres1,max_iter1,verbose)

%---------------------run the algorithm------------------------------
%1xN cell vectors
pie_all = init_paras{1};
o_all = init_paras{2};
sigma_all = init_paras{3};
indivec_all = init_paras{4};

LL_all = [];
paras = cell(1,6); 
if nargout > 2
    p_states = [];
    aCN = zeros(1,length(o_all));
    segments = cell(1,length(o_all));
end

for i = 1:length(o_all)
    %1x1 cell
    init_paras(1) = pie_all(i);
    init_paras(2) = o_all(i);
    init_paras(3) = sigma_all(i);
    init_paras(4) = indivec_all(i);
    
    [LL,pie,o,sigma,iterations] = estimate_paras(init_paras,depend_table,thres1,max_iter1,verbose);
        
    LL_all = [LL_all LL(end)];
    paras{1} = [paras{1} {pie}];
    paras{2} = [paras{2} {o}];
    paras{3} = [paras{3} {sigma}];
    paras{4} = [paras{4} init_paras(4)];
    
    if nargout > 2
        [temp,aCN(i),segments{i}] = process_results(depend_table);
        p_states = [p_states temp];
    end

    if verbose
        disp('--------------- screening report -----------------')
        disp(['run ' num2str(i) ' done, iterations:' num2str(iterations)]);
        disp(['sigma:' num2str(sigma) ', o:' num2str(o) ', LL:' num2str(LL(end),'%5.1f')]);
        disp('--------------- screening report -----------------')
    end
    
end