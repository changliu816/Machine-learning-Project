function [yHat] = NaiveBayesian(XTrain, YTrain, XTest)

[m, n] = size(XTrain);
m0 = sum(YTrain == 0);
m1 = sum(YTrain == 1);

X0 = XTrain(YTrain == 0, :);
X1 = XTrain(YTrain == 1, :);

mu0 = mean(X0);
se0 = std(X0);
Sigma0 = cov(X0);
invSig0 = inv(Sigma0);
det0 = det(Sigma0);
p0 = m0/m;

mu1 = mean(X1);

se1 = std(X1);
Sigma1 = cov(X1);
invSig1 = inv(Sigma1);
det1 = det(Sigma1);

[mTest, n] = size(XTest);
yHat = zeros(mTest, 1);

for i = 1:mTest 
    X = XTest(i, :);
    Px0 = -sum((X-mu0).*(X-mu0)/2./se0./se0) - sum(log(se0));
    Px1 = -sum((X-mu1).*(X-mu1)/2./se1./se1) - sum(log(se1));
    %Px0 = -log(det0)/2 - 0.5 * (X-mu0) * invSig0 * (X-mu0)';
    %Px1 = -log(det1)/2 - 0.5 * (X-mu1) * invSig1 * (X-mu1)';
    if (Px0 + log(p0) > Px1 + log(1 - p0))
        yHat(i) = 0;
    else 
        yHat(i) = 1;
    end
end

end
