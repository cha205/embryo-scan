function runClassicalBaseline(options)
%RUNCLASSICALBASELINE Time cell divisions with classical image processing.
%   runClassicalBaseline() finds t2 to t5 in every held-out test video without
%   machine learning, saves the results to runs/timelapse/classical-matlab/
%   and calls reportClassicalBaseline. For each video it locates the embryo
%   (classical.findEmbryo), measures frame-to-frame change inside it
%   (classical.motionEnergy) and turns motion spikes into division times
%   (classical.detectDivisions). Positions and sizes are in pixels of the
%   original 500 x 500 frames.
%
%   Options:
%     Split  "test" (default) or "val"
%     Limit  only the first N videos, without updating the report
%
%   Example, from the repository root:
%     addpath("matlab"); runClassicalBaseline(Limit=3)

arguments
    options.Split (1, 1) string {mustBeMember(options.Split, ["val", "test"])} = "test"
    options.Limit (1, 1) double {mustBePositive} = Inf
end

SCALE = 0.5;          % frames are analysed at 250 x 250
TYPICAL_RADIUS = 50;  % embryo radius at SCALE, used when no circle is found

p = classical.projectPaths();
deepFile = fullfile(p.deepRun, "predictions-" + options.Split + ".csv");
if ~isfile(deepFile)
    error("EmbryoScan:missingPredictions", ...
        "No deep-model predictions at %s.\nRun: python scripts/timelapse/train_stages.py --eval-only --split %s", ...
        deepFile, options.Split);
end
if ~isfolder(p.runDir)
    mkdir(p.runDir);
end
logFile = fullfile(p.runDir, "run-" + options.Split + ".log");
if isfile(logFile)
    delete(logFile);
end
diary(logFile);
stopLog = onCleanup(@() diary("off"));

frameTable = classical.readFrameTable();
videoSplit = readtable(p.videoSplit, "TextType", "string");
deep = readtable(deepFile, "TextType", "string");
videos = videoSplit.video(videoSplit.split == options.Split);
videos = videos(1:min(end, options.Limit));

n = numel(videos);
summary = cell(n, 1);
energyRows = cell(n, 1);
fprintf("Classical baseline on %d %s videos\n", n, options.Split);
timer = tic;
for i = 1:n
    id = videos(i);
    labels = frameTable(frameTable.video == id, :);
    [frames, hours] = classical.loadVideo(id, frameTable, Scale=SCALE);

    [center, radius, located] = classical.findEmbryo(median(frames(:, :, 1:min(20, end)), 3));
    if isnan(radius)
        radius = TYPICAL_RADIUS;
    end
    energy = classical.motionEnergy(frames, center, radius);
    detected = classical.detectDivisions(energy, hours);

    annotated = classical.onsetsFromPhases(labels.phase, labels.hours);
    predicted = deep(deep.video == id, :);
    model = classical.onsetsFromPhases(predicted.decoded, predicted.hours);

    info = table(id, located, center(1) / SCALE, center(2) / SCALE, radius / SCALE, ...
        'VariableNames', ["video", "located_by", "center_x", "center_y", "radius"]);
    summary{i} = [info, onsetColumns(annotated, "annotated"), ...
        onsetColumns(detected, "classical"), onsetColumns(model, "deep")];
    energyRows{i} = table(repmat(id, numel(hours), 1), labels.frame, hours, energy, ...
        'VariableNames', ["video", "frame", "hours", "energy"]);

    fprintf("[%3d/%d] %-12s t2 %5.1f h (annotated %5.1f)  t5 %5.1f h (annotated %5.1f)  %4.0f s\n", ...
        i, n, id, detected.t2, annotated.t2, detected.t5, annotated.t5, toc(timer));
end

writetable(vertcat(summary{:}), fullfile(p.runDir, "divisions-" + options.Split + ".csv"));
writetable(vertcat(energyRows{:}), fullfile(p.runDir, "energy-" + options.Split + ".csv"));
fprintf("Saved per-video results to %s\n", p.runDir);

if isfinite(options.Limit)
    fprintf("Quick check only: report not updated (run without Limit to update it).\n");
else
    reportClassicalBaseline(Split=options.Split);
end
end

function columns = onsetColumns(onsets, source)
%ONSETCOLUMNS One-row table of division times named like classical_t2.
columns = struct2table(onsets);
columns.Properties.VariableNames = source + "_" + string(columns.Properties.VariableNames);
end
