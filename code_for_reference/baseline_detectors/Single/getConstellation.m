function cons = getConstellation(ModType)

    binaryCase = combn([0 1], ModType);
    cons = zeros(1,length(binaryCase));   
    for i = 1:length(binaryCase)
       
        cons(i) = GrayMapQAM(binaryCase(i,:), ModType);
        
    end
end