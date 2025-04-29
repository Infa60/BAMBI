function btkCutAcquisition(h, startAt, numFrames)
%BTKCUTACQUISITION Cut the acquistion and adapt the events' frame/time.
% 
%  BTKCUTACQUISITION(H, STARTAT) keeps all the frames from the frame STARTAT.
%  The acquistion is represented by the handle H, obtained by the use of a 
%  btk* function.
%
%  BTKCUTACQUISITION(H, STARTAT, NUMFRAMES) keeps NUMFRAMES frames starting
%  from the frame STARTAT.

%  Author: A. Barr�
%  Copyright 2009-2012 Biomechanical ToolKit (BTK).

ff = btkGetFirstFrame(h);
lf = btkGetLastFrame(h);
if (nargin == 2)
    numFrames = lf - ff + 1 - startAt;
end
if ((startAt < ff) || startAt > lf)
    error('btk:CutAcquisiton','Invalid index.');
elseif (numFrames > lf - startAt + 1)
    error('btk:CutAcquisiton','Incorrect number of frames specified.')
elseif (numFrames == 0) % Clear all
    error('btk:CutAcquisiton','Due to the mechanism used in BTK, it is not possible to remove all the frames. Contact the developers if you really need it.')
end
% Data to keep
% - Point
pidx = (startAt:startAt+numFrames-1)-ff+1;
[points, pointsInfo] = btkGetPoints(h);
pv = struct2array(points);
pv = pv(pidx,:);
rv = struct2array(pointsInfo.residuals);
rv = rv(pidx,:);
% - Analog
snpf = btkGetAnalogSampleNumberPerFrame(h);
aidx = (((startAt-ff)*snpf):(startAt-ff+numFrames)*snpf-1)+1;
av = btkGetAnalogsValues(h);
av = av(aidx,:);
% Resizing
btkSetFrameNumber(h, numFrames);
% Storing modifications
for i = 1:size(pv,2)/3
    btkSetPoint(h, i, pv(:,(i-1)*3+1:i*3), rv(:,i));
end
if isempty(av)==0
    btkSetAnalogsValues(h, av);
end
% Set the first frame.
%btkSetFirstFrame(h, startAt, 1); % 1: Modify also the events' frame/time