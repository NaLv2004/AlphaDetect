function Opt = K_Best(R, sym, z, k) 

conSize = length(sym);
currentSurvive=[];
for step=1:length(z)
 expandedPath=[repmat(1:conSize,1,max(size(currentSurvive,2),1));kron(currentSurvive,ones(1,conSize))];
 PED=sum((z(end-step+1:end)- R(end-step+1:end,end-step+1:end)*sym(expandedPath)).^2,1);
 [~,choice]=mink(PED,k);
 currentSurvive=expandedPath(:,choice);
end
Opt= sym(currentSurvive(:,1))';
end