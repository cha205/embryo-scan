function p = projectPaths()
%PROJECTPATHS Data and output locations, matching embryoscan/paths.py.
%   Set the EMBRYOSCAN_DATA environment variable to use another data folder.

root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
dataDir = string(getenv("EMBRYOSCAN_DATA"));
if dataDir == ""
    dataDir = fullfile(root, "data");
end

p.root = string(root);
p.gomezFrames = fullfile(dataDir, "raw", "gomez2022", "embryo_dataset");
p.frameTable = fullfile(dataDir, "processed", "timelapse", "frames.csv");
p.videoSplit = fullfile(p.root, "splits", "timelapse-videos.csv");
p.deepRun = fullfile(p.root, "runs", "timelapse", "stages-resnet34");
p.runDir = fullfile(p.root, "runs", "timelapse", "classical-matlab");
p.reportDir = fullfile(p.root, "reports", "timelapse");
p.figureDir = fullfile(p.root, "reports", "figures");
end
