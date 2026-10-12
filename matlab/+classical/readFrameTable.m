function frameTable = readFrameTable()
%READFRAMETABLE Frame table written by scripts/timelapse/prepare_data.py.
%   frameTable = classical.readFrameTable() returns one row per frame with
%   video, frame (1-based), hours after fertilization and phase (index into
%   PHASES in embryoscan/timelapse/labels.py, or -1 before annotations start).

p = classical.projectPaths();
opts = detectImportOptions(p.frameTable, "TextType", "string");
opts.SelectedVariableNames = ["video", "frame", "hours", "phase"];
frameTable = readtable(p.frameTable, opts);
end
