session = IEEGSession('I521_A0012_D001', 'smcafee', 'smc_ieeglogin.bin');
session.openDataSet('I521_A0012_D002');
session.openDataSet('I521_A0012_D003');

session.openDataSet('I521_A0013_D001');
session.openDataSet('I521_A0013_D002');
session.openDataSet('I521_A0013_D003');

session.openDataSet('I521_A0014_D001');
session.openDataSet('I521_A0014_D002');
session.openDataSet('I521_A0014_D003');

for i=1:3
    trainInd = (i-1)*3+1;
    labelInd = (i-1)*3+2;
    testInd = (i-1)*3+3;
    
    endTime = session.data(trainInd).rawChannels(1).get_tsdetails.getEndTime;
    endPt = ceil((endTime)/1e6* ...
                 session.data(trainInd).sampleRate);    
    [nChannels, ~] = size(session.data(trainInd).channelLabels);
    trainData{i} = session.data(trainInd).getvalues(1:endPt, 1: ...
                                                           nChannels);
    labelData{i} = session.data(labelInd).getvalues(1:endPt, 1:5);
    
    endTime = session.data(testInd).rawChannels(1).get_tsdetails.getEndTime;
    endPt = ceil((endTime)/1e6* ...
                 session.data(testInd).sampleRate)'
    testData{i} = session.data(testInd).getvalues(1:endPt, 1:nChannels);
end