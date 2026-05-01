function b = symboltobits(symbol, length,QAM)

Tx_B_mat = zeros(length, log2(QAM)/2);
switch(QAM)
    case 4
        for t0=1:length
            if symbol(t0)<=0
                Tx_B_mat(t0) = 0;
            else
                Tx_B_mat(t0) = 1;
            end
        end
        
    case 16        
        symbol = symbol * 0.5 + 1.5;
        for t0=1:length
            if(round(symbol(t0))<1)
                Tx_B_mat(t0,:)=[0 1];
            elseif (round(symbol(t0))<2)
                Tx_B_mat(t0,:)=[0 0];
            elseif (round(symbol(t0))<3)
                Tx_B_mat(t0,:)=[1 0];
            else
                Tx_B_mat(t0,:)=[1 1];
            end
        end  
    
    case 64            
        symbol = symbol * 0.5 + 3.5;
        for t0=1:length
            if(round(symbol(t0))<1)
                Tx_B_mat(t0,:)=[ 0 1 1];
            elseif(round(symbol(t0))<2)
                Tx_B_mat(t0,:)=[ 0 1 0];
            elseif(round(symbol(t0))<3)
                Tx_B_mat(t0,:)=[ 0 0 0];
            elseif(round(symbol(t0))<4)
                Tx_B_mat(t0,:)=[ 0 0 1];
            elseif(round(symbol(t0))<5)
                Tx_B_mat(t0,:)=[ 1 0 1];
            elseif(round(symbol(t0))<6)
                Tx_B_mat(t0,:)=[ 1 0 0];
            elseif(round(symbol(t0))<7)
                Tx_B_mat(t0,:)=[ 1 1 0];
            else
                Tx_B_mat(t0,:)=[ 1 1 1];
            end
        end     
        
    case 256        
        symbol = symbol * 0.5 + 15/2;
        for t0=1:length
            if(round(symbol(t0))<1)
                Tx_B_mat(t0,:)=[ 0 1 1 1];
            elseif(round(symbol(t0))<2)
                Tx_B_mat(t0,:)=[ 0 1 1 0];
            elseif(round(symbol(t0))<3)
                Tx_B_mat(t0,:)=[ 0 1 0 0];
            elseif(round(symbol(t0))<4)
                Tx_B_mat(t0,:)=[ 0 1 0 1];
            elseif(round(symbol(t0))<5)
                Tx_B_mat(t0,:)=[ 0 0 0 1];
            elseif(round(symbol(t0))<6)
                Tx_B_mat(t0,:)=[ 0 0 0 0];
            elseif(round(symbol(t0))<7)
                Tx_B_mat(t0,:)=[ 0 0 1 0];
            elseif(round(symbol(t0))<8)
                Tx_B_mat(t0,:)=[ 0 0 1 1];
            elseif(round(symbol(t0))<9)
                Tx_B_mat(t0,:)=[ 1 0 1 1];
            elseif(round(symbol(t0))<10)
                Tx_B_mat(t0,:)=[ 1 0 1 0];
            elseif(round(symbol(t0))<11)
                Tx_B_mat(t0,:)=[ 1 0 0 0];
            elseif(round(symbol(t0))<12)
                Tx_B_mat(t0,:)=[ 1 0 0 1];
            elseif(round(symbol(t0))<13)
                Tx_B_mat(t0,:)=[ 1 1 0 1];
            elseif(round(symbol(t0))<14)
                Tx_B_mat(t0,:)=[ 1 1 0 0];
            elseif(round(symbol(t0))<15)
                Tx_B_mat(t0,:)=[ 1 1 1 0];
            else
                Tx_B_mat(t0,:)=[ 1 1 1 1];
            end
        end        
end
Tx_B_matH=Tx_B_mat';
b=Tx_B_matH(:);
