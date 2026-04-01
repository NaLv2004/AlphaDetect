function symest = AMP(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum)

G = H'*H;
Gtilde = eye(2*TxAntNum) - diag(1./diag(G))*G;
g = diag(G)/(2*RxAntNum);
gtilde = 1./diag(G);
yMF = H'*RxSymbol;
yMFtilde = gtilde.*yMF;

shat   = zeros(2*TxAntNum,1);
tau_s=ones(2*TxAntNum,1);
tau_p=g' * tau_s;
tau_z= (tau_p  + Nv)*gtilde;
theta_tau_s = 0.5;
theta_tau_z =0.5;

z=yMFtilde;

for k=1:iterNum

    % -- compute distance
    input_symM = abs(z - sym').^2;
    expo = min(input_symM,[],2)-input_symM;

    % -- compute pdf
    pdf = exp(expo./tau_z);

    % -- compute weight constant
    w = pdf./sum(pdf,2);

    % -- compute mean (F)
    shat_old=shat;
    shat = w*sym;

    % -- compute variance (G)
    tau_s = sum(w.*(abs(sym'-shat).^2),2);

    if any(isnan(pdf))|any(isinf(pdf))
        warning('LAMA_MeanVar: NaN or inf OCCURRED');
    end
    tau_p_old=tau_p;
    tau_p  = theta_tau_s * ( g' * tau_s ) + (1 - theta_tau_s) * tau_p;

    v = tau_p / (tau_p_old + Nv) * ( z - shat_old );

    tau_z = theta_tau_z * (tau_p + Nv) * gtilde + (1-theta_tau_z) * tau_z;

    z = yMFtilde + Gtilde * shat + v;
end

[~,index]=min(abs(z-sym'),[],2);
symest=sym(index);



