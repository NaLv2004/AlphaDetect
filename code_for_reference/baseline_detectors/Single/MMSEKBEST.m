function Opt = MMSEKBEST(R, sym, z, k,TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,delta,iterNum) 

S_mmse = (H'*H+Nv*eye(2*TxAntNum)) \ H'*RxSymbol;
conSize = length(sym);
currentSurvive=[];
for step=1:length(z)
 expandedPath=[repmat(1:conSize,1,max(size(currentSurvive,2),1));kron(currentSurvive,ones(1,conSize))];
 PED=sum((z(end-step+1:end)- R(end-step+1:end,end-step+1:end)*sym(expandedPath)).^2,1);
 MMSE_dis= sum((S_mmse(end-step+1:end)-sym(expandedPath)).^2,1);
 [~,choice]=mink(PED+MMSE_dis,k);
 currentSurvive=expandedPath(:,choice);
end
Opt= sym(currentSurvive(:,1))';
end