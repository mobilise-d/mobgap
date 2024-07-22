dataFolder = '/home/arne/Documents/repos/private/mobilised_tvs_data/tvs_dataset';

% MS 4038 wrong weight
folder = fullfile(dataFolder, "MS", "4038", "Free-living");
file = fullfile(folder, "infoForAlgo.mat");
load(file);
infoForAlgo.TimeMeasure1.Weight = 55;
save(file, "infoForAlgo")

% 1093/1095 wrong dominant hand
file = fullfile(dataFolder, "HA", "1093", "Free-living", "infoForAlgo.mat");
load(file);
infoForAlgo.TimeMeasure1.Handedness = 'L';
save(file, "infoForAlgo")

file = fullfile(dataFolder, "HA", "1095", "Free-living", "infoForAlgo.mat");
load(file);
infoForAlgo.TimeMeasure1.Handedness = 'L';
save(file, "infoForAlgo")