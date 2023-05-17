function [LL,pie,o,sigma,nrIterations] = ...
    estimate_paras(init_paras,depend_table,thresh,max_iter,verbose)
% 02/12/2022 by Zhenhua

previous_loglik = -inf;
converged = 0;
num_iter = 1;
LL = [];

pie = init_paras{1};
o = init_paras{2};
sigma = init_paras{3};

while (num_iter <= max_iter) && ~converged
    % perform EM algorithm
    [loglik,pie_u,o_u,sigma_u] = ...
        update_paras(pie,o,sigma,depend_table);
    
    converged = em_converged_m(loglik,previous_loglik,verbose,thresh);
    
    % update parameters
    if init_paras{4}(1)
        pie = pie_u;
    end
    if init_paras{4}(2) %update o
        o = o_u;
    end
    if init_paras{4}(3) %update sigma
        sigma = sigma_u;
    end
    
    if verbose
        disp(['sigma:' num2str(sigma) ', o:' num2str(o)]);
%         disp(['pie:' num2str(pie')]);
        fprintf(1, 'iteration %d, loglik = %f\n', num_iter, loglik);
    end
    
    num_iter =  num_iter + 1;
    previous_loglik = loglik;
    LL = [LL loglik];
end
nrIterations = num_iter - 1;

end

%--------------------------------------------------------------------------
function [loglik,pie_u,o_u,sigma_u] = ...
    update_paras(pie,o,sigma,depend_table)
global data_lrc_sep
global data_seg_sep
global gamma_sep

numex = length(data_lrc_sep);

%-----------------------E step-----------------------------
gamma_sep = cell(1,numex);
loglik = 0;

for ex = 1:numex
    
    % conditional probabilities
    log_obslik = get_log_obslik(data_lrc_sep{ex},data_seg_sep{ex},o,sigma,depend_table);
    % Forward and Backward algorithm
    [gamma,current_ll] = get_post_probs(pie,log_obslik);
    
    loglik = loglik + current_ll;
    gamma_sep{ex} = gamma;
end

%-----------------------M step-----------------------------
%update sigma
sigma_u = update_sigma(o,sigma,depend_table);

%update o
o_u = update_o(o,depend_table);

%update pie
pie_u = update_pie(depend_table);

end


function [gamma,ll] = get_post_probs(pie,log_obslik)

gamma = zeros(size(log_obslik));
ll = 0;
for k = 1:size(log_obslik,2)
    tmp = zeros(1,size(log_obslik,1));
    m = max(log_obslik(:,k));
    for s = 1:size(log_obslik,1)
        tmp(s) = exp(log_obslik(s,k)-m);
    end
    denominator = tmp*pie;
    for s = 1:size(log_obslik,1)
        gamma(s,k) = pie(s)*exp(log_obslik(s,k)-m)/denominator;
    end
    ll = ll+log(denominator*exp(m)+eps);
end

end

%--------------------------------------------------------------------------
function sigma_u = update_sigma(o,sigma,depend_table)

global data_lrc_sep
global data_seg_sep
global gamma_sep

numex = length(data_lrc_sep); % number of chromosomes
tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries
mu_l = log2(Y/2)+o;
    
% numerators = zeros(1,length(sigma));
% denominators = zeros(1,length(sigma));
numerator = 0;
denominator = 0;

for ex = 1:numex
    obs_lrc = data_lrc_sep{ex};
    data_seg = data_seg_sep{ex};
    post_probs = gamma_sep{ex}(1:length(Y),:);
    for i = 1:length(Y)
        for k = 1:size(post_probs,2)
            s_indx = data_seg(k,1);
            e_indx = data_seg(k,2);
            numerator = numerator+post_probs(i,k)*(sum((obs_lrc(s_indx:e_indx)-mu_l(i)).^2));
            denominator = denominator+post_probs(i,k)*(e_indx-s_indx+1);
        end
    end
%     denominator = denominator+length(obs_lrc);
end

sigma_u = sqrt(numerator/denominator);
if isnan(sigma_u)
    sigma_u = sigma;
end

end

%--------------------------------------------------------------------------
function o_u = update_o(o,depend_table)
global data_lrc_sep
global data_seg_sep
global gamma_sep

numex = length(data_lrc_sep); % number of chromosomes
tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries

tmp1 = log2(Y/2);

numerator = 0;
denominator = 0;

for ex = 1:numex
    obs_lrc = data_lrc_sep{ex};
    data_seg = data_seg_sep{ex};
    post_probs = gamma_sep{ex}(1:length(Y),:);
    for i = 1:length(Y)
        for k = 1:size(post_probs,2)
            s_indx = data_seg(k,1);
            e_indx = data_seg(k,2);
            numerator = numerator+post_probs(i,k)*sum((obs_lrc(s_indx:e_indx)-tmp1(i)));
            denominator = denominator+post_probs(i,k)*(e_indx-s_indx+1);
        end
    end
%     denominator = denominator+length(obs_lrc);
end

o_u = numerator/denominator;
if isnan(o_u)
    o_u = o;
end

end

%--------------------------------------------------------------------------
function pie_u = update_pie(depend_table)
global gamma_sep

tv = depend_table(:,2) == 1;
Y = depend_table(tv,3); %copy numbers of different entries
pie_u = zeros(length(Y),1);

numex = length(gamma_sep);
for ex = 1:numex
    post_probs = gamma_sep{ex}(1:length(Y),:);
    pie_u = pie_u+sum(post_probs,2);
end

pie_u = pie_u/sum(pie_u);

end

