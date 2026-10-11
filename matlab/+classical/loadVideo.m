function [frames, hours] = loadVideo(videoId, frameTable, options)
%LOADVIDEO Read one time-lapse video as a grayscale image stack.
%   [frames, hours] = classical.loadVideo(videoId, frameTable) reads every
%   frame listed for videoId in frameTable (data/processed/timelapse/
%   frames.csv) and returns an H-by-W-by-T uint8 stack plus the hours after
%   fertilization of each frame. A missing or unreadable frame repeats the
%   previous one, and every frame is resized to the first one's size.
%
%   Options:
%     Scale  resize factor applied to every frame (default 1)

arguments
    videoId (1,1) string
    frameTable table
    options.Scale (1,1) double {mustBePositive} = 1
end

p = classical.projectPaths();
rows = frameTable(frameTable.video == videoId, :);
hours = rows.hours;

% Frame files are named ..._RUN<frame>.jpeg, sometimes one folder deeper.
files = dir(fullfile(p.gomezFrames, videoId, "**", "*.jpeg"));
if isempty(files)
    error("EmbryoScan:missingFrames", ...
        "No frames for %s under %s.\nRun: python scripts/timelapse/download_data.py", ...
        videoId, p.gomezFrames);
end
names = string({files.name});
runNumber = double(extractBetween(names, "RUN", ".jpeg"));
lookup = zeros(max([runNumber, rows.frame']), 1);
lookup(runNumber) = 1:numel(files);

% A few source JPEGs are truncated; imread still decodes what is there.
warningState = warning("off", "MATLAB:imagesci:jpg:libraryMessage");
restoreWarnings = onCleanup(@() warning(warningState));

frames = [];
frameSize = [];
previous = [];
for t = 1:height(rows)
    img = readFrame(files, lookup, rows.frame(t));
    if isempty(img)
        img = previous;  % missing or truncated source JPEG: repeat the last frame
    else
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        if isempty(frameSize)
            frameSize = round(size(img, [1 2]) * options.Scale);
        end
        if ~isequal(size(img, [1 2]), frameSize)
            img = imresize(img, frameSize);
        end
    end
    if isempty(img)
        continue  % nothing readable yet
    end
    if isempty(frames)
        % frames before the first readable one copy it
        frames = repmat(img, 1, 1, height(rows));
    end
    frames(:, :, t) = img;
    previous = img;
end
end

function img = readFrame(files, lookup, frame)
%READFRAME One source frame, or [] when it is missing or unreadable. Some
%   timestamp files list a frame 0 that has no image.
img = [];
if frame >= 1 && lookup(frame) > 0
    try
        img = imread(fullfile(files(lookup(frame)).folder, files(lookup(frame)).name));
    catch
        % truncated source JPEG
    end
end
end
