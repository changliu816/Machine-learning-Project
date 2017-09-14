function [ yHat ] = Perception( Xtrain, Ytrain, Xtest)
[mtrain, ntrain] = size(Xtrain);
wT = zeros(1, ntrain);
i = 0; 
numIter = 100;
b = 0;
while ( i < numIter )
    misCount = 0;
    for j = 1:mtrain
        xj = Xtrain(j, :)';
        xj = xj/max(xj)-0.5;
        if (Ytrain(j) * (wT * xj + b) <= 0)
            misCount = misCount+1;
            wT = wT + Ytrain(j) * xj';
            b = b + Ytrain(j);
        end
    end
    if misCount == 0
        break;
    end
    i = i + 1;
end

[mtest, ntest] = size(Xtest);
yHat = zeros(mtest, 1);
for j = 1:mtest
    xj = Xtest(j, :)';
    xj = xj/max(xj)-0.5;
    if (wT * xj + b) > 0
        yHat(j) = 1;
    end
end
end

