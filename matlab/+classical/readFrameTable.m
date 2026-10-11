function frameTable = readFrameTable()
%READFRAMETABLE One row per video frame, from the prepared time-lapse data.
%   frameTable = classical.readFrameTable() reads data/processed/timelapse/
%   frames.csv (written by scripts/timelapse/prepare_data.py) with columns
%   video, frame (1-based), hours after fertilization and phase (index into
%   PHASES in embryoscan/timelapse/labels.py, or -1 before the first
%   annotation).

p = classical.projectPaths();
opts = detectImportOptions(p.frameTable, "TextType", "string");
opts.SelectedVariableNames = ["video", "frame", "hours", "phase"];
frameTable = readtable(p.frameTable, opts);
end
