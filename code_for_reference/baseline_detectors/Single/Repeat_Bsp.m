function symest = Repeat_Bsp(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum,TxSymbol_real,norm_f)


limit=1;
symLLR=RD_BsP_df1dm1_Det(TxAntNum,RxAntNum,slen,RxSymbol,H,Nv,sym,delta,iterNum);
[~,Indice] = max(symLLR);
symest = sym(Indice);


addloop=0;
while sum((RxSymbol-H*symest-limit*sqrt(Nv))>0)~=0 && addloop<=5
    addloop=addloop+1;

    Tx_fake=-sign(symest)*norm_f*2.*randi(size(sym,1)/2,[2*TxAntNum 1]);
    

    Rx_fake=H*Tx_fake;
    Rx_add=RxSymbol+Rx_fake;


    test_temp=TxSymbol_real+Tx_fake;

    symLLR=RD_BsP_df1dm1_Det(TxAntNum,RxAntNum,slen,Rx_add,H,Nv,sym,delta,iterNum);
    [~,Indice] = max(symLLR);
    symest_add = sym(Indice);

    symest_new=symest_add-Tx_fake;

    if sum((RxSymbol-H*symest_new-limit*sqrt(Nv))>0)<sum((RxSymbol-H*symest-limit*sqrt(Nv))>0)
        symest=symest_new;
    end
end
if sum((RxSymbol-H*symest-limit*sqrt(Nv))>0)==0 && sum(sign(abs(int8((symest-TxSymbol_real)/norm_f))))~=0
    a=1;
end


end
