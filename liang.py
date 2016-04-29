import pdb
import scipy.io as sio
import numpy as np
import scipy.signal as sig
import math
from scipy import interpolate
from sklearn import linear_model
from sklearn import feature_selection
from sklearn import svm
from sklearn import pipeline as pipe

def Bandpass(lowcut, highcut, trans_width, fs):
    numtaps = 100
    edges = [0, lowcut - trans_width,
             lowcut, highcut,
             highcut + trans_width, 0.5*fs]
    taps = sig.remez(numtaps, edges, [0, 1, 0], Hz=fs)
    return taps
    
def CreateBands(data, fs):
    length, channels = data.shape
    ret = np.zeros((length, 3*channels))
    filt_1_60 = Bandpass(1, 60, 0.5, fs)
    filt_60_100 = Bandpass(60, 100, 3, fs)
    filt_100_200 = Bandpass(100, 200, 5, fs)

    ind = 0
    for i in range(channels):
        ret[:, ind] = sig.lfilter(filt_1_60, [1], data[:, i])
        ind = ind + 1
        ret[:, ind] = sig.lfilter(filt_60_100, [1], data[:, i])
        ind = ind + 1        
        ret[:, ind] = sig.lfilter(filt_100_200, [1], data[:, i])
        ind = ind + 1
    return ret

def WinSize(fs):
    return int(round(fs*40e-3))
        
def NumWins(data, fs):
    length, _ = data.shape
    return int(math.floor(length/WinSize(fs)))

def SquareVoltage(data, fs):
    length, width = data.shape
    
    nWins = NumWins(data, fs)
    winSize = WinSize(fs)
    ret = np.zeros((nWins, width))

    for i in range(nWins):
        fst = int(i*winSize)
        lst = int(fst+winSize)
        tmp = data[fst:lst, :]
        tmp = np.square(tmp)
        tmp = np.sum(tmp, axis=0)
        ret[i, :] = tmp
    
    return ret

def CreateFeatures(squares, fs, num_lag):
    length, width = squares.shape
    numRows = length - num_lag + 1
    ret = np.zeros((numRows, num_lag*width))
    
    for i in range(numRows):
        for j in range(num_lag):
            for k in range(width):
                ret[i, j*width+k] = squares[i+j, k]
    # pdb.set_trace();
    ret = np.pad(ret, ((num_lag-1, 0), (0, 0)), 'constant')
    return ret

d = sio.loadmat('data.mat')
labelData = d['labelData']
trainData = d['trainData']
testData = d['testData']
fs = d['fs']
num_lag = 25
pred = np.empty((3, 1), dtype=object)

for i in range(3):
    data = CreateBands(trainData[0, i], fs[i])
    squares = SquareVoltage(data, fs[i])
    feats = CreateFeatures(squares, fs[i], num_lag)

    labels = sig.decimate(labelData[0, i], int(0.04*fs[i].item()), axis=0)

    dataTest = CreateBands(testData[0, i], fs[i])
    squaresTest = SquareVoltage(dataTest, fs[i])
    featsTest = CreateFeatures(squaresTest, fs[i], num_lag)
    
    (nPts, _) = testData[0, i].shape
    duration_ECoG = (nPts-1)/float(fs[i].item());
    
    for j in range(5):
        clf = svm.SVR(kernel='linear')
        clf.fit(feats, labels[:, j])
        predLabels = clf.predict(featsTest)

        x = np.linspace(0, duration_ECoG, predLabels.size)
        tck = interpolate.splrep(x, predLabels, s=0)
        xnew = np.linspace(0, duration_ECoG, nPts)
        ynew = interpolate.splev(xnew, tck, der=0)
        predLabels = ynew

    pred[i, 0] = predLabels
sio.savemat('liang.mat', {'predicted_dg': pred})
