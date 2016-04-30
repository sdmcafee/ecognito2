if ~exist('session')
    %Subject 1 ECoG train
    session = IEEGSession('I521_A0012_D001', 'smcafee', 'smc_ieeglogin.bin');
    duration_ECoG = session.data(1).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(1).sampleRate; % sampling rate, Hz
    train_ECoG_1 = session.data(1).getvalues(1:duration_ECoG/1e6*fs+1,1);
    % Pull all EEG data
    train_ECoG_1 = zeros(62,duration_ECoG/1e6*fs+1);
    for i = 1:62
        train_ECoG_1(i,:) = session.data(1).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
    %Subject 1 Glove test
    session.openDataSet('I521_A0012_D002');
    duration_glove = session.data(2).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(2).sampleRate; % sampling rate, Hz
    train_glove_1 = zeros(5,duration_glove/1e6*fs+1);
    for i = 1:5
        train_glove_1(i,:) = session.data(2).getvalues(1:duration_glove/1e6*fs+1,i);
    end
    
    %Subject 1 ECoG test
    session.openDataSet('I521_A0012_D003');
    duration_ECoG = session.data(3).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(3).sampleRate; % sampling rate, Hz
    % Pull all EEG data
    test_ECoG_1 = zeros(62,duration_ECoG/1e6*fs+1);
    for i = 1:62
        test_ECoG_1(i,:) = session.data(3).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
    %% Subject 2
    %subject 2 ECoG train
    session.openDataSet('I521_A0013_D001');
    duration_ECoG = session.data(4).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(4).sampleRate; % sampling rate, Hz
    % Pull all EEG data
    train_ECoG_2 = zeros(48,duration_ECoG/1e6*fs+1);
    for i = 1:48
        train_ECoG_2(i,:) = session.data(4).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
    %Subject 2 Glove train
    session.openDataSet('I521_A0013_D002');
    duration_glove = session.data(5).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(5).sampleRate; % sampling rate, Hz
    train_glove_2 = zeros(5,duration_glove/1e6*fs+1);
    for i = 1:5
        train_glove_2(i,:) = session.data(5).getvalues(1:duration_glove/1e6*fs+1,i);
    end
    
    %Subject 2 ECoG test
    session.openDataSet('I521_A0013_D003');
    duration_ECoG = session.data(6).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(6).sampleRate; % sampling rate, Hz
    % Pull all EEG data
    test_ECoG_2 = zeros(48,duration_ECoG/1e6*fs+1);
    for i = 1:48
        test_ECoG_2(i,:) = session.data(6).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
    
    %% Subject 3
    %subject 3 ECoG train
    session.openDataSet('I521_A0014_D001');
    duration_ECoG = session.data(7).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(7).sampleRate; % sampling rate, Hz
    % Pull all EEG data
    train_ECoG_3 = zeros(64,duration_ECoG/1e6*fs+1);
    for i = 1:64
        train_ECoG_3(i,:) = session.data(7).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
    %Subject 3 Glove train
    session.openDataSet('I521_A0014_D002');
    duration_glove = session.data(8).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(8).sampleRate; % sampling rate, Hz
    train_glove_3 = zeros(5,duration_glove/1e6*fs+1);
    for i = 1:5
        train_glove_3(i,:) = session.data(8).getvalues(1:duration_glove/1e6*fs+1,i);
    end
    
    %Subject 3 ECoG test
    session.openDataSet('I521_A0014_D003');
    duration_ECoG = session.data(9).rawChannels(1).get_tsdetails.getDuration;
    fs = session.data(9).sampleRate; % sampling rate, Hz
    % Pull all EEG data
    test_ECoG_3 = zeros(64,duration_ECoG/1e6*fs+1);
    for i = 1:64
        test_ECoG_3(i,:) = session.data(9).getvalues(1:duration_ECoG/1e6*fs+1,i);
    end
    
end



%% decimate glove
dec_glove_1 = zeros(5,length(train_glove_1)/50);
for i = 1:5
    dec_glove_1(i,:) = decimate(train_glove_1(i,:)',50);
end

dec_glove_2 = zeros(5,length(train_glove_2)/50);
for i = 1:5
    dec_glove_2(i,:) = decimate(train_glove_2(i,:)',50);
end

dec_glove_3 = zeros(5,length(train_glove_3)/50);
for i = 1:5
    dec_glove_3(i,:) = decimate(train_glove_3(i,:)',50);
end

%%
[idxTrain, idxTest] = crossvalind('HoldOut', 6200, .3);


%% Feature Classification
% Feature Extracting
% Set data to extract from
x = train_ECoG_2;
channel = min(size(x));

fs = 1000;
winLen = .1;
winDisp = .05;

% Defining anon feat funcs
avgVolt = @(x) mean(x);
E = @(x) sum(x.^2);
A = @(x) sum(abs(x));
NumWins = @(x, fs, winLen, winDisp) floor((length(x) - winLen*fs)/(winDisp*fs))+1;
norm = @(x) bsxfun(@rdivide, x-diag(mean(x,2))*ones(size(x)),std(x,0,2));

% % extract mean voltage
% mask = ones(1,window)/window;
% mean_volt = conv(x(1,:),mask,'valid');

volt_feat = zeros(channel,NumWins(x,fs,winLen,winDisp));
for i = 1:channel
    volt_feat(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,avgVolt);
end

output = tsmovavg(x,'s',winLen*fs,2);

% var_feat = zeros(channel,NumWins(x,fs,winLen,winDisp));
% for i = 1:channel
%     var_feat(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) var(x));
% end

% Extract Features: SPECTROGRAM METHOD
noverlap = fs .* winDisp;
window = fs .* winLen;
nfft = 1024;

f5_15_1 = zeros(NumWins(x,fs,winLen,winDisp),channel);
f20_25_1 = zeros(NumWins(x,fs,winLen,winDisp),channel);
f75_115_1 = zeros(NumWins(x,fs,winLen,winDisp),channel);
f125_160_1 = zeros(NumWins(x,fs,winLen,winDisp),channel);
f160_175_1 = zeros(NumWins(x,fs,winLen,winDisp),channel);
for i = 1:channel
    [s,f,~] = spectrogram(x(i,:), window, noverlap, nfft, fs);
    mag = abs(s);
    % 5 to 15 Hz
    f5_15_1(:,i) = mean(mag(f>5 & f<15,:));
    % 20 to 25 Hz
    f20_25_1(:,i) = mean(mag(f>20 & f<25,:));
    % 75 to 115 Hz
    f75_115_1(:,i) = mean(mag(f>75 & f<115,:));
    % 125 to 160 Hz
    f125_160_1(:,i) = mean(mag(f>125 & f<160,:));
    % 160 to 175 Hz
    f160_175_1(:,i) = mean(mag(f>160 & f<175,:));
end



% Linear model
y = [volt_feat; f5_15_1'; f20_25_1'; f75_115_1'; f125_160_1'; f160_175_1'];
y = norm(y);
num_lag = 4;
X = ones(max(size(y))+1-num_lag,min(size(y))*num_lag+1);
% v = num classes, M = num time bins, n = time bins before
for v = 1:min(size(y))
    for M = 1:(max(size(y))+1-num_lag)
        for n = 1:num_lag
            X(M,((v-1)*num_lag+1)+n) = y(v,M+n-1);
        end
    end
end
X = padarray(X, [num_lag 0], 'pre');
Xtrain= X(idxTrain,:);
Xtest = X(idxTest,:);

%% Making beta matrix
Xtrain = Xtrain(:,feats_picked);

beta = (Xtrain'*Xtrain)\(Xtrain'*dec_glove_2(:,idxTrain)');

%% Scaling back up for submission
% interpolating spline
Xtest = Xtest(:,feats_picked);

y_hat = Xtest*beta;
duration_ECoG = 147499000;
sub3dg = spline(linspace(0,duration_ECoG,length(y_hat')),y_hat',linspace(0,duration_ECoG,length(x)));

%% package
predicted_dg = {sub1dg';sub2dg';sub3dg'};

%% feature selection

%fun = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));
% fun = @(XT,yT,Xt,yt) (rmse((regress(yT, XT)'*Xt')'), yt);
% inmodel = sequentialfs(fun,X,dec_glove_1(1,:)');

im = zeros(5,min(size(X)));

[~,~,~,im(1,:),~,~,~] = stepwisefit(Xtrain,dec_glove_2(1,idxTrain)');
[~,~,~,im(2,:),~,~,~] = stepwisefit(Xtrain,dec_glove_2(2,idxTrain)');
[~,~,~,im(3,:),~,~,~] = stepwisefit(Xtrain,dec_glove_2(3,idxTrain)');
[~,~,~,im(4,:),~,~,~] = stepwisefit(Xtrain,dec_glove_2(4,idxTrain)');
[~,~,~,im(5,:),~,~,~] = stepwisefit(Xtrain,dec_glove_2(5,idxTrain)');

feats_picked = sum(im) > 1;


%% ensemble subject 1
mdl1_1 = fitglm(X,(dec_glove_1(1,:)'-min(dec_glove_1(1,:))+.01),'Distribution','gamma');
mdl1_2 = fitglm(X,dec_glove_1(2,:)');
mdl1_3 = fitglm(X,dec_glove_1(3,:)');
mdl1_4 = fitglm(X,dec_glove_1(4,:)');
mdl1_5 = fitglm(X,dec_glove_1(5,:)');


dg1_1 = predict(mdl1_1,X);
dg1_2 = predict(mdl1_2,X);
dg1_3 = predict(mdl1_3,X);
dg1_4 = predict(mdl1_4,X);
dg1_5 = predict(mdl1_5,X);
y_hat = [dg1_1, dg1_2, dg1_3, dg1_4, dg1_5];

%% ensemble subject 2
mdl2_1 = fitensemble(y',dec_glove_2(1,:)','LSBoost',100,'Tree');
mdl2_2 = fitensemble(y',dec_glove_2(2,:)','LSBoost',100,'Tree');
mdl2_3 = fitensemble(y',dec_glove_2(3,:)','LSBoost',100,'Tree');
mdl2_4 = fitensemble(y',dec_glove_2(4,:)','LSBoost',100,'Tree');
mdl2_5 = fitensemble(y',dec_glove_2(5,:)','LSBoost',100,'Tree');


dg2_1 = predict(mdl2_1,y');
dg2_2 = predict(mdl2_2,y');
dg2_3 = predict(mdl2_3,y');
dg2_4 = predict(mdl2_4,y');
dg2_5 = predict(mdl2_5,y');
dg2 = [dg2_1, dg2_2, dg2_3, dg2_4, dg2_5];

%% ensemble subject 3
mdl3_1 = fitensemble(y',dec_glove_3(1,:)','LSBoost',100,'Tree');
mdl3_2 = fitensemble(y',dec_glove_3(2,:)','LSBoost',100,'Tree');
mdl3_3 = fitensemble(y',dec_glove_3(3,:)','LSBoost',100,'Tree');
mdl3_4 = fitensemble(y',dec_glove_3(4,:)','LSBoost',100,'Tree');
mdl3_5 = fitensemble(y',dec_glove_3(5,:)','LSBoost',100,'Tree');


dg3_1 = predict(mdl3_1,y');
dg3_2 = predict(mdl3_2,y');
dg3_3 = predict(mdl3_3,y');
dg3_4 = predict(mdl3_4,y');
dg3_5 = predict(mdl3_5,y');
y_hat = [dg3_1, dg3_2, dg3_3, dg3_4, dg3_5];

%%
mdl1_1 = lasso(X,dec_glove_1(1,:)');
mdl1_2 = lasso(X,dec_glove_1(2,:)');
mdl1_3 = lasso(X,dec_glove_1(3,:)');
mdl1_4 = lasso(X,dec_glove_1(4,:)');
mdl1_5 = lasso(X,dec_glove_1(5,:)');

mdl2_1 = lasso(X,dec_glove_2(1,:)');
mdl2_2 = lasso(X,dec_glove_2(2,:)');
mdl2_3 = lasso(X,dec_glove_2(3,:)');
mdl2_4 = lasso(X,dec_glove_2(4,:)');
mdl2_5 = lasso(X,dec_glove_2(5,:)');

mdl3_1 = lasso(X,dec_glove_3(1,:)');
mdl3_2 = lasso(X,dec_glove_3(2,:)');
mdl3_3 = lasso(X,dec_glove_3(3,:)');
mdl3_4 = lasso(X,dec_glove_3(4,:)');
mdl3_5 = lasso(X,dec_glove_3(5,:)');



dg1_1 = X*mdl1_1(:,1);
dg1_2 = X*mdl1_2(:,1);
dg1_3 = X*mdl1_3(:,1);
dg1_4 = X*mdl1_4(:,1);
dg1_5 = X*mdl1_5(:,1);
dg1 = [dg1_1, dg1_2, dg1_3, dg1_4, dg1_5];

dg2_1 = X*mdl2_1(:,1);
dg2_2 = X*mdl2_2(:,1);
dg2_3 = X*mdl2_3(:,1);
dg2_4 = X*mdl2_4(:,1);
dg2_5 = X*mdl2_5(:,1);
dg2 = [dg2_1, dg2_2, dg2_3, dg2_4, dg2_5];

dg3_1 = X*mdl3_1(:,1);
dg3_2 = X*mdl3_2(:,1);
dg3_3 = X*mdl3_3(:,1);
dg3_4 = X*mdl3_4(:,1);
dg3_5 = X*mdl3_5(:,1);
dg3 = [dg3_1, dg3_2, dg3_3, dg3_4, dg3_5];


