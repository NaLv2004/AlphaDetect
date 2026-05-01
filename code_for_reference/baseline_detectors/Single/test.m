load H_all.mat;


indice=nchoosek(1:32,2);
res=zeros(size(H_all,3),1);
for i=1:size(H_all,3)
HMat=H_all(:,:,i);
Hreal = [real(HMat), -imag(HMat); imag(HMat), real(HMat)];
A=Hreal(indice(:,1),:);
B=Hreal(indice(:,2),:);

k=acos(sum(A.*B,2)./sqrt(sum(A.^2,2))./sqrt(sum(B.^2,2)));
res(i)=min(k);
end

