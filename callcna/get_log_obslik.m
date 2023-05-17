function log_obslik = get_log_obslik(data_lrc,data_seg,o,sigma,depend_table)

K = size(data_seg,1);

tv_S = depend_table(:,2) == 1;
Y = depend_table(tv_S,3); %vector of copy numbers of different entries
mu_l = log2(Y/2)+o;

S = sum(tv_S);
log_obslik = zeros(S,K);

for i = 1:S
    for k = 1:K
        s_indx = data_seg(k,1);
        e_indx = data_seg(k,2);
        obslik_lrc = eval_pdf_lrc(data_lrc(s_indx:e_indx),mu_l(i),sigma);
         if Y(i) == 2
             obslik_lrc = 1.05*obslik_lrc;
         end
         if Y(i) == 1
             obslik_lrc = 1.1*obslik_lrc;
         end
        ll = sum(log(obslik_lrc+eps));
        log_obslik(i,k) = ll;
    end
end

end

