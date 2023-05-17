function paras = main(paras_0,depend_table,thres_EM,max_iter,verbose)
%02/12/2022 by Zhenhua
%--------------------------- screening -------------------------
global clamp_thres
global mc_w
clamp_thres = 1-1e-3;
mc_w = 0.8;

thres_del = 0.009;
paras_0 = init_paras(paras_0,depend_table,0);
[LL,paras,p_states,aCN] = screening...
    (paras_0,depend_table,thres_EM,max_iter,verbose);

tv_del = depend_table(depend_table(:,2)~=0,3)<1;
p_total_del = sum(p_states(tv_del,:),1);
disp(num2str(p_total_del))
disp(num2str(aCN))

tv = (p_total_del<thres_del) & aCN < 4.5;
if ~any(tv)
    tv = ~tv;
end
candi_indx = find(tv);
candi_ll = LL(tv);
candi_acn = aCN(tv);

[temp,I] = max(candi_ll);
best_indx = candi_indx(I);

[temp,indxs] = sort(candi_ll,'descend');
scores = 1;
acns = candi_acn(indxs(1));
for i = 2:length(candi_indx)
    j = indxs(i-1);
    pre_indxs = indxs(1:i-1);
    k = indxs(i);
    if abs(candi_ll(j)-candi_ll(k)) <= 2 && abs(candi_acn(j)-candi_acn(k)) <= 0.1
        continue;
    end
    score = (candi_acn(pre_indxs)-candi_acn(k)+eps)./(candi_ll(pre_indxs)-candi_ll(k)+eps);
    scores = [scores score(end)];
    acns = [acns candi_acn(k)];
    if sum(score >= 0.1) == i-1
        best_indx = candi_indx(k);
    end
end

% tmp = (aCN(best_indx)-aCN(tv)+eps)./(LL(best_indx)-LL(tv)+eps);
disp(['acn: ' num2str(acns)])
disp(['score: ' num2str(scores)])
disp(['ll: ' num2str(candi_ll(indxs))])

% Now, use the optimal parameters to call CNA
%-------------------------------------------------------------------
paras_0 = init_paras(paras,depend_table,best_indx);
[temp,paras] = screening(paras_0,depend_table,5*thres_EM,20,verbose);
%-------------------------------------------------------------------


function paras = init_paras(paras_0,depend_table,best_indx)
%this function is used to initialize/process parameters for training,
%best_indx is used to indicate which parameter configuration (ususally
%multiple generated in previous screening procedure) are selected. If
%best_indx equals 0, parameters will be initialized
global var_l

paras = cell(1,5);
%parameter initialization
if best_indx == 0
    %---o---
    if isempty(paras_0{2})
        o_0 = [-1 -0.6 -0.3 0];
        %o_0 = [-1.3 -1 -0.6 -0.3 0];
%         o_0 = [0];
    else
        o_0 = paras_0{2};
    end
    
    N = length(o_0);
    paras{2} = mat2cell(o_0,1,ones(1,N));
    
    %---sigma--- 
    if isempty(paras_0{3})
        sigma_0 = sqrt(var_l);
    else
        sigma_0 = paras_0{3};
    end
    paras{3} = repmat({sigma_0},1,N);

    %---pi---
    if isempty(paras_0{1})
        S = sum(depend_table(:,2) ~= 0);
        pie_0 = ones(S,1)/S;
    else
        pie_0 = paras_0{1};
    end
    paras{1} = repmat({pie_0},1,N);
    
    %---indicator vector---
    if isempty(paras_0{4}) %indicator vector: '1' for update '0' fixed
        adj_all = ones(1,3);
    else
        adj_all = paras_0{4};
    end
    paras{4} = repmat({adj_all},1,N);
else %parse the results from previous screening
    for i = 1:length(best_indx)
        %--pi--
        paras{1} = [paras{1} paras_0{1}(best_indx(i))];
         %--o--
        paras{2} = [paras{2} paras_0{2}(best_indx(i))];
        %--sigma--
        paras{3} = [paras{3} paras_0{3}(best_indx(i))];
        %--indicator vector--
        paras{4} = [paras{4} paras_0{4}(best_indx(i))];
    end
end
