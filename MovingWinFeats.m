function y = MovingWinFeats(x, fs, winLen, winDisp, featFn)

xLen = length(x);

NumWins = floor((xLen - winLen.*fs)/(winDisp.*fs))+1;

y = zeros(1,NumWins);

winStart = 1;
winEnd = winLen .* fs;

for i = 1:NumWins
    xWin = x(winStart:winEnd);
    y(i) = featFn(xWin);
    winStart = winStart + winDisp;
    winEnd = winEnd + winDisp;
end

end