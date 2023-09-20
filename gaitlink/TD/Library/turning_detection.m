function [Turns]= turning_detection(wz, fc, fs, turnThres)

%% INPUTS
% wz = angular velocity around the vertical axis in deg/s
% fc = cut-off frequency for the filter
% fs = sampling frequency of the signal
% turnThres = turn angles to be considered

%% OUTPUTS
% Turns is a structure that contains:
% .Magnitude = magnitude of the detected turns [deg]
% .TurnDur = beginning and end of the detected turns [frames]

    wz_filt =low_pass_filter(wz,fc,fs,4);
    
    %Abs(filter angular velocity) in deg/s
    wz_filtabs =abs(wz_filt);
    
    %Integration of wz: yaw angle in deg
    t = 1/fs:1/fs:(length(wz)/fs);
    yaw_angle =cumtrapz(t,wz_filt);

%% Segmentation gyro signals adapted from El-Gohary et al.,2004 (https://doi.org/10.3390/s140100356)    
    [Turns] = gyro_sgm(yaw_angle, wz_filt, wz_filtabs, t, fs, turnThres);
    
	Turns.Yaw = yaw_angle;
end

%% Function
function FiltSign = low_pass_filter(sign,Fc,Fs,n)
    Wn=Fc/(Fs/2);
    [b,a]=butter(n,Wn,'low');
    FiltSign=filtfilt(b,a,sign);
end

