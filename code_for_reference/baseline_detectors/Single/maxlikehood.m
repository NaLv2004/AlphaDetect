function Opt = maxlikehood(RxSymbol,Hreal,cons)

allMin=Inf;

val= repmat({cons'},size(Hreal,2)/2,1);
[x{1 : numel(val)}] = ndgrid(val{:});
ret = reshape(cat(numel(val), x{:}), [], numel(val))';
for i=1:size(ret,2)
        temp=[ret;repmat(ret(:,i),1,size(ret,2))];
        [currentMin,choice]=min(sum((RxSymbol-Hreal*temp).^2,1));
        if currentMin<allMin
            allMin=currentMin;
            Opt=temp(:,choice);
        end
end

end

