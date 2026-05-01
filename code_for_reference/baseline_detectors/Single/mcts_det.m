function [res,kbestCount,PEDCount] = mcts_det(R, sym, z,Nv,TxSymbol_real)
[~,answer]=min(abs(TxSymbol_real-sym'),[],2);


%playout=ones(100,1)*30;
playout=[40,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
cCut=1.0;    % 超参数
maxPEDAllowed=1.5*Nv*length(R);
maxRestart=5;


gameLength=length(z);
conSize = length(sym);


childs={[]};   % 子节点表
nodeVisitTime=[1];%访问次数表
score=[0];       %得分表
nodeData=[0];  % 节点内容表
parent=[0];
fullExpanded=[0];
restart=0;


kbestCount=0;
PEDCount=[];


while restart<maxRestart
    movePlayed=[1];

    for layer=1:gameLength



        for play=1:playout(layer)   % 走子尝试
            currentNode=movePlayed(end);
            accumulatedNode=movePlayed;  accumulatedNode(1)=[];


            while fullExpanded(currentNode)

                candidates=childs{currentNode};

                vParent=nodeVisitTime(currentNode);
                vChilds=nodeVisitTime(candidates);

                %qualityScore=score(candidates)./nodeVisitTime(candidates);
                qualityScore=score(candidates);
                visitScore=cCut*sqrt(log(vParent)./vChilds);

                UCT=qualityScore+visitScore;

                [~,choice]=max(UCT);                      % 走子
                currentNode=candidates(choice);
                accumulatedNode=[accumulatedNode;currentNode];

            end

            step=length(accumulatedNode);

            if step~=gameLength

                candidates=childs{currentNode};
                playLeft=setdiff(1:conSize,nodeData(candidates));


                candidatePED=sum((z(end-step:end)- R(end-step:end,end-step:end)* reshape(sym([playLeft;repmat(flip(nodeData(accumulatedNode)),1,length(playLeft))]),[length(accumulatedNode)+1,length(playLeft)])).^2,1);

                candidatePED(candidatePED>maxPEDAllowed)=Inf;
                legalChild=sum(candidatePED~=Inf);
                fullExpanded(currentNode)=legalChild<=1;

                if legalChild>=1
                    [~,choice]=min(candidatePED);
                    newChildData=playLeft(choice);



                    Opt = Half_K_Best(R, sym, z, 1,[newChildData;flip(nodeData(accumulatedNode))]);
                    kbestCount=kbestCount+1-((1+step)*step)/((1+gameLength)*gameLength);
                    childPED=sum((z-R*sym(Opt)).^2);
                    PEDCount=[PEDCount;childPED];

                    %childPED=sum((z-R*sym([randi([1,conSize],gameLength-step-1,1);newChildData;nodeData(flip(accumulatedNode))])).^2,1);

                    scoreBackuped=1/(1+childPED);
                    %scoreBackuped=-childPED;

                    childsAddedNum=gameLength-step;

                    childs{currentNode}=[childs{currentNode} length(childs)+1 ] ;
                    parent=[parent;currentNode;[length(childs)+1:length(childs)+childsAddedNum-1]'];
                    childs=[childs;num2cell([length(childs)+2:length(childs)+childsAddedNum]');{[]}];
                    nodeData=[nodeData;flip(Opt(1:end-step))];
                    nodeVisitTime=[nodeVisitTime;ones(childsAddedNum,1)];
                    fullExpanded=[fullExpanded;zeros(childsAddedNum,1)];
                    score=[score;repmat(scoreBackuped,childsAddedNum,1)];
                else
                    scoreBackuped=score(currentNode);
                end

            else
                scoreBackuped=score(accumulatedNode(end));
            end

            accumulatedNode=[movePlayed(1);accumulatedNode];
            currentScore=score(accumulatedNode);
            score(accumulatedNode)=(currentScore>=scoreBackuped).*currentScore+(currentScore<scoreBackuped).*scoreBackuped;
            nodeVisitTime(accumulatedNode)=nodeVisitTime(accumulatedNode)+1;



        end

        legalPlay=childs{movePlayed(end)};
        [~,choice]=max(score(legalPlay));
        currentRoot=legalPlay(choice);
        movePlayed=[movePlayed;currentRoot];

    end
    res=sym(nodeData(flip(movePlayed(2:end))));
    finalPED=sum((z-R*res).^2);
    if finalPED>maxPEDAllowed
        restart=restart+1;
    else
        restart=Inf;
    end
end
res=sym(nodeData(flip(movePlayed(2:end))));
err=res-TxSymbol_real;
end


function Opt = Half_K_Best(R, sym, z, k,currentSurvive) % 酷炫版的kbest
conSize = length(sym);
for step=(length(currentSurvive)+1):length(z)
    expandedPath=[repmat(1:conSize,1,size(currentSurvive,2));kron(currentSurvive,ones(1,conSize))];
    PED=sum((z(end-step+1:end)- R(end-step+1:end,end-step+1:end)*sym(expandedPath)).^2,1);
    [~,choice]=mink(PED,k);
    currentSurvive=expandedPath(:,choice);
end
Opt=currentSurvive(:,1);
end
