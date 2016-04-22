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


% Defining anon feat funcs
avgVolt = @(x) mean(x);
NumWins = @(x, fs, winLen, winDisp) floor((length(x) - winLen*fs)/(winDisp*fs))+1;


%% Subject 1 Classifying
% Feature Extracting

fs = 1000;
winLen = .1;
winDisp = .05;

noverlap = fs .* winDisp;
window = fs .* winLen;
nfft = 1024;

% Set data to extract from
x = test_ECoG_1;
channel = min(size(x));

% extract mean voltage
mask = ones(1,window)/window;
mean_volt = conv(x(1,:),mask,'valid');


volt_feat_1 = zeros(channel,NumWins(x,fs,winLen,winDisp));
for i = 1:channel
    volt_feat_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,avgVolt);
end



%% TO-DO: extract frequency bands

% SPECTROGRAM METHOD
winLen = .1;
winDisp = .05;
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

test_feats_1 = [volt_feat_1./max(max(abs(volt_feat_1))); f5_15_1'; f20_25_1'; f75_115_1'; f125_160_1'; f160_175_1'];

test_feats_1 = [volt_feat_1./max(max(abs(volt_feat_1))); f5_15_1./max(max(abs(f5_15_1)));...
    f75_115_1./max(max(abs(f75_115_1)));...
    f125_160_1./max(max(abs(f125_160_1))); f160_175_1./max(max(abs(f160_175_1)))];

train_feats_1 = padarray(train_feats_1, [0 1], 'pre');





% decimate glove
dec_glove_3 = zeros(5,length(train_glove_3)/50);
for i = 1:5
    dec_glove_3(i,:) = decimate(train_glove_3(i,:)',50);
end

sub1dec = sub1decnet(train_feats_1);


% interpolating spline
sub3dg = spline(linspace(0,duration_ECoG,length(sub3')),sub3',linspace(0,duration_ECoG,length(train_glove_3)));

% Matrix to make finger prediction
finger_guess = zeros(1,length(guess));
for i=1:length(guess)
    if guess(1,i) < 0.9
        [~,finger] = max(guess(2:end,i));
        finger_guess(i) = finger;
    end
end


% Linear model

X = ones(6199,310);

% v = num classes, M = num time bins, n = time bins before

for v = 1:310
    for M = 1:6199
        for n = 1:3
            X(M,((v-1)*3+1)+n) = train_feats_1(M+n-1,v);
        end
    end
end

X = train_feats_1';
beta = (X'*X)\(X'*dec_glove_1');




%% Subject 2 Classifying
% prep for nnets
tr_glove_2 = zeros(6,length(train_glove_2));
for i = 1:length(tr_glove_2)
    finger = train_glove_2(i);
    if (finger < 0) || (finger > 5)
        continue
    end
    tr_glove_2(finger+1,i) = 1;
end

%% Subject 3 Classifying
% prep for nnets
tr_glove_3 = zeros(6,length(train_glove_3));
for i = 1:length(tr_glove_2)
    finger = train_glove_3(i);
    if (finger < 0) || (finger > 5)
        continue
    end
    tr_glove_3(finger+1,i) = 1;
end












%% Placeholder for garbage

% 5 to 15 Hz
f5_15_1 = zeros(62,NumWins(x,fs,winLen,winDisp));
for i = 1:62
    f5_15_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) meanPower(x,fs,5,15));
end

% % 20 to 25 Hz
% f20_25_1 = zeros(62,NumWins(x,fs,winLen,winDisp));
% for i = 1:62
%     f20_25_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) meanPower(x,fs,20,25));
% end

% 75 to 115 Hz
f75_115_1 = zeros(62,NumWins(x,fs,winLen,winDisp));
for i = 1:62
    f75_115_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) meanPower(x,fs,75,115));
end

% 125 to 160 Hz
f125_160_1 = zeros(62,NumWins(x,fs,winLen,winDisp));
for i = 1:62
    f125_160_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) meanPower(x,fs,125,160));
end

% 160 to 175 Hz
f160_175_1 = zeros(62,NumWins(x,fs,winLen,winDisp));
for i = 1:62
    f160_175_1(i,:) = MovingWinFeats(x(i,:),fs,winLen,winDisp,@(x) meanPower(x,fs,160,175));
end
