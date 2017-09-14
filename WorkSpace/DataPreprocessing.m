%Diagnostic = importdata('wdbc.data');
%Prognostic = importdata('wpbc.data');

Prognostic = xlsread('wpbc.xlsx','Prognostic_data');
[m,n] = size(Prognostic);
attributes = 4:n;
XTrain = Prognostic(:, attributes);
A = (Prognostic(:, 2) == 0);
B = (Prognostic(:, 2) == 1);
%scatter(Prognostic(:, 3), Prognostic(:, 2))

x = zeros(1, 13);
y = zeros(1, 13);

scatter(Prognostic(A, 4), Prognostic(A, 16), 'o');
hold on;
scatter(Prognostic(B, 4), Prognostic(B, 16), '+');
hold off;

for i = 1:m 
    t = ceil(Prognostic(i, 3)/10);
    if (Prognostic(i, 2) == 1)
        x(t) = x(t) + 1;
    else 
        y(t) = y(t) + 1;
    end
end
%plot(x./(x+y))
%% find highly correlated features
cor = corrcoef(XTrain);
for i = 1:n-3
    for j = i+1:n-3
        if cor(i, j) > 0.9
            fprintf('%d, %d\n', i, j);
        end
    end
end

%% Normalize Data
TrainMax = max(XTrain);
for i = 1:n-3
    XTrain(:, i) = XTrain(:, i)/TrainMax(i) - 0.5;
end

%% affine attributes
attributes = [4,10,13,16,19,22,25,28,31,34,35];

%% Pure Training
XTrain = Prognostic(:, attributes);
YTrain = Prognostic(:, 2);
yHat = NaiveBayesian(XTrain, YTrain, XTrain);
display(sum(yHat == YTrain));

display(sum(yHat == 0 & YTrain == 0));
display(sum(yHat == 0 & YTrain == 1));
display(sum(yHat == 1 & YTrain == 0));
display(sum(yHat == 1 & YTrain == 1));

%% N-Folder
chunck = 10;
chuncksize = ceil(m/chunck);
ordering = randperm(m);
Prognostic = Prognostic(ordering, :);

yRes = [];
for i = 1:chunck
    startIdx = (i-1)*chuncksize+1;
    endIdx = min(m, i * chuncksize);
    XTest = Prognostic(startIdx: endIdx, attributes);
    YTest = Prognostic(startIdx: endIdx, 2);
    
    XTrain = [Prognostic(1: startIdx-1, attributes); Prognostic(endIdx+1:end, attributes)];
    YTrain = [Prognostic(1: startIdx-1, 2); Prognostic(endIdx+1:end, 2)];
    
    yHat = NaiveBayesian(XTrain, YTrain, XTest);
    yRes = [yRes; yHat];
end
yTest = Prognostic(:, 2);
display(sum(yRes == 0 & yTest == 0));
display(sum(yRes == 0 & yTest == 1));
display(sum(yRes == 1 & yTest == 0));
display(sum(yRes == 1 & yTest == 1));    


%% PCA 


%% qqplot
%A = Prognostic(:, 2) == 0;
%B = Prognostic(:, 2) == 1;
titles = {'Radius'; 'Texture'; 'Perimeter'; 'Area'; 'Smoothness'; 'Compactness'; 'Concavity'; 'Concave points'; 'Symmetry'; 'Fractal dimension'};
for i = 4:3:33
   subplot(4,3,(i-1)/3);
   qqplot(Prognostic(:, i));
   title(titles((i-1)/3));
   xlabel('Standard Normal');
   ylabel(titles((i-1)/3));
end