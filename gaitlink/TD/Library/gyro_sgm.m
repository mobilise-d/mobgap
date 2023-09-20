function [Turns] = gyro_sgm(yaw_angle, wz_filt, wz_filtabs, time, fs, turnThres)

MaxVal = 15; % Find maxima of absolute value
[hpeaks,locpeak]=findpeaks(wz_filtabs,'MinPeakHeight',MaxVal);
n = size(wz_filtabs,1);

% figure
% subplot(2,1,1)
% title('Filtered Angular Velocity (deg/s)');
% plot(time, wz_filtabs,'Color',[0.7 0.7 0.7],'DisplayName','wz filtered [deg/s]')
% hold on
% plot(locpeak/fs,hpeaks,'*r');

CrossVal = 5 ;
zci = @(v) find(v(:).*circshift(v(:),[-1 0]) <= 0);
zx = zci(wz_filt - CrossVal);

% identify 5deg/s crossings preceding and following each maximum 
x1 = [];
x2 = [];
for i = 1:length(locpeak)
    bef = zx<locpeak(i);
    if isempty(find(bef,1))
        x1 = [x1;1];
    else
        bef = zx(bef);
        x1 = [x1;bef(end)];
    end
    aft = zx>locpeak(i);
    if isempty(find(aft,1))
        x2 = [x2;length(wz_filt)];
    else
        aft = zx(aft);
        x2 = [x2;aft(1)];
    end    
end

allChanges = unique(sort([x1;x2]));

% subplot(2,1,2)
% plot(time,yaw_angle(1:n))
% hold on

% Turn definitions
StartStop = zeros(size(allChanges,1)-1,2);
TurnM = zeros(size(allChanges,1)-1,1);
TurnDur = zeros(size(allChanges,1)-1,1);
j = 1;
for i = 2:length(allChanges)
    StartStop(j,:) = [allChanges(i-1), allChanges(i)];
    TurnM(j) =  yaw_angle(allChanges(i))-yaw_angle(allChanges(i-1));
    TurnDur(j) = (allChanges(i)- allChanges(i-1))/fs;
    j = j+1;
end

i = 1;
while i < length(TurnDur)
    if TurnDur(i)< 0.05 && (sign(TurnM(i))==sign(TurnM(i+1)))
        StartStop(i,:) = [StartStop(i,1), StartStop(i+1,2)];
        StartStop(i+1,:) = [];
        TurnM(i) =  yaw_angle(StartStop(i,2))-yaw_angle(StartStop(i,1));
        TurnM(i+1) = [];
        TurnDur(i) = (StartStop(i,2)-StartStop(i,1))/fs;
        TurnDur(i+1) = [];        
    end
    i = i+1;
end

Check = true(size(TurnDur));
for i = 1:length(TurnDur)
    if TurnDur(i)> 10 || (TurnDur(i) < 0.5)
        Check(i) = false(1);
    end
    if abs(TurnM(i))<turnThres
        Check(i) = false(1);
    end
end

TurnM = TurnM(Check);
TurnDur = TurnDur(Check);
StartStop = StartStop(Check,:);

% for i = 1:length(TurnDur)
%     line([StartStop(i,1)/fs StartStop(i,1)/fs],get(gca,'ylim'),'Color',[0 0 1]);
%     line([StartStop(i,2)/fs StartStop(i,2)/fs],get(gca,'ylim'),'Color',[0 0 1]);
% end

Turns.Magnitude = TurnM;
Turns.TurnDur = StartStop;
end