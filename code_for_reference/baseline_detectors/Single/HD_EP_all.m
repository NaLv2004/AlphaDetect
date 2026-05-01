function out = HD_EP_all(input, s)
out = zeros(length(input),1);
for i = 1:length(input)
    ep = abs(input(i) - s);
    [~,k] = min(ep);
    out(i) = s(k);
end
end