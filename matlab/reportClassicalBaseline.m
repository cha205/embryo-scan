function reportClassicalBaseline(options)
%REPORTCLASSICALBASELINE Score the classical baseline and draw its figures.
%   reportClassicalBaseline() reads the per-video results saved by
%   runClassicalBaseline and compares the classical division times with the
%   embryologist's and the deep model's on the same videos. Writes
%   reports/timelapse/classical-baseline-<split>.md and, for the test
%   split, two figures in reports/figures/:
%
%     timelapse-classical-motion-energy.png  one typical video, step by step
%     timelapse-classical-vs-deep.png        divisions timed within 1 h
%
%   The deep model's numbers are recomputed from its saved predictions with
%   the same rules, so both methods are scored identically.

arguments
    options.Split (1, 1) string {mustBeMember(options.Split, ["val", "test"])} = "test"
end

p = classical.projectPaths();
split = options.Split;
results = readtable(fullfile(p.runDir, "divisions-" + split + ".csv"), "TextType", "string");

score = scoreTimings(results);
agreement = categoryAgreement(results);
withFigures = split == "test";

report = fullfile(p.reportDir, "classical-baseline-" + split + ".md");
writelines(reportLines(results, score, agreement, split, withFigures), report);
fprintf("Wrote %s\n", report);

if withFigures
    if ~isfolder(p.figureDir)
        mkdir(p.figureDir);
    end
    energy = readtable(fullfile(p.runDir, "energy-" + split + ".csv"), "TextType", "string");
    plotTypicalVideo(results, energy, fullfile(p.figureDir, "timelapse-classical-motion-energy.png"));
    plotComparison(score, height(results), fullfile(p.figureDir, "timelapse-classical-vs-deep.png"));
    fprintf("Wrote figures to %s\n", p.figureDir);
end
end

%% Scoring

function score = scoreTimings(results)
%SCORETIMINGS Timing error of each method per division, against the
%   embryologist. A division a method missed counts as a miss in the
%   within-1 h and within-2 h rates; median errors are over those it found.
phases = ["t2", "t3", "t4", "t5"];
rows = cell(numel(phases), 1);
for k = 1:numel(phases)
    truth = results.("annotated_" + phases(k));
    has = ~isnan(truth);
    row = struct("phase", phases(k), "videos", nnz(has));
    for method = ["classical", "deep"]
        err = results.(method + "_" + phases(k))(has) - truth(has);
        row.(method + "_found") = mean(~isnan(err));
        row.(method + "_median_abs") = median(abs(err), "omitnan");
        row.(method + "_within_1h") = mean(abs(err) <= 1);
        row.(method + "_within_2h") = mean(abs(err) <= 2);
    end
    rows{k} = struct2table(row);
end
score = vertcat(rows{:});
end

function agreement = categoryAgreement(results)
%CATEGORYAGREEMENT How often each method's Meseguer category matches the one
%   from the annotated times, over videos where both can be scored.
truth = categoriesOf(results, "annotated");
for method = ["classical", "deep"]
    guess = categoriesOf(results, method);
    scored = truth ~= "X" & guess ~= "X";
    agreement.(method) = mean(truth(scored) == guess(scored));
    agreement.(method + "_videos") = nnz(scored);
end
end

function category = categoriesOf(results, source)
%CATEGORIESOF Meseguer category of every video from one source's times.
category = strings(height(results), 1);
for i = 1:height(results)
    onsets = struct();
    for phase = ["t2", "t3", "t4", "t5"]
        onsets.(phase) = results.(source + "_" + phase)(i);
    end
    category(i) = classical.meseguerCategory(onsets);
end
end

%% Report

function lines = reportLines(results, score, agreement, split, withFigures)
%REPORTLINES The markdown report, one string per line.
n = height(results);
lines = [
    "# Classical baseline: division timing without machine learning"
    ""
    sprintf("The same %d held-out %s videos as [divisions-%s.md](divisions-%s.md), timed by a " + ...
        "hand-built MATLAB pipeline (Image Processing Toolbox, nothing trained) instead of the " + ...
        "deep model. Code: [matlab/](../../matlab).", n, split, split, split)
    ""
    sprintf("1. **Find the embryo**: circular Hough transform (`imfindcircles`) on the median of " + ...
        "the first 20 frames found it in %d of %d videos; in the rest, the most textured region " + ...
        "(cells are grainy, the culture medium is smooth) stood in.", ...
        nnz(results.located_by == "hough"), n)
    "2. **Motion energy**: the mean change between consecutive frames inside the embryo, in units " + ...
        "of the video's own noise. A cell splitting in two changes many pixels at once, so " + ...
        "divisions show up as spikes."
    "3. **Rules from embryology**: t2 is the biggest spike between 18 and 38 h, t3 the first spike " + ...
        "5 to 18 h later, t4 a spike within 4 h of t3 (none means both cells split together), t5 " + ...
        "the first spike 6 to 20 h after t4."
    ""
    "Timing error per division (detected minus the embryologist's time). A division a method " + ...
        "missed counts as a miss in the within-1 h and within-2 h columns; median errors are over " + ...
        "the divisions it found. The deep model's numbers are recomputed from its saved " + ...
        "predictions with the same rules."
    ""
    timingTable(score)
    ""
    sprintf("- **Meseguer 2011 category** (A+ ... D-) matches the one from the embryologist's " + ...
        "times for %.0f%% of %d videos with the classical times and %.0f%% of %d with the deep " + ...
        "model's.", 100 * agreement.classical, agreement.classical_videos, ...
        100 * agreement.deep, agreement.deep_videos)
    ];
if withFigures
    lines = [lines
        ""
        "![Motion energy for one typical video](../figures/timelapse-classical-motion-energy.png)"
        ""
        "![Divisions timed within 1 h, classical vs deep model](../figures/timelapse-classical-vs-deep.png)"];
end
end

function lines = timingTable(score)
%TIMINGTABLE Markdown table comparing both methods, one row per division.
lines = [
    "| division | videos | found by classical | median error, classical | median error, deep | " + ...
        "within 1 h, classical | within 1 h, deep | within 2 h, classical | within 2 h, deep |"
    "|---|---|---|---|---|---|---|---|---|"
    ];
for k = 1:height(score)
    s = score(k, :);
    lines(end + 1, 1) = sprintf("| %s | %d | %.0f%% | %.2f h | %.2f h | %.0f%% | %.0f%% | %.0f%% | %.0f%% |", ...
        s.phase, s.videos, 100 * s.classical_found, s.classical_median_abs, s.deep_median_abs, ...
        100 * s.classical_within_1h, 100 * s.deep_within_1h, ...
        100 * s.classical_within_2h, 100 * s.deep_within_2h); %#ok<AGROW>
end
end

%% Figures

function plotTypicalVideo(results, energy, file)
%PLOTTYPICALVIDEO Motion energy, detected divisions and frames for the video
%   whose classical timing error is the median among fully detected videos.
c = palette();
phases = ["t2", "t3", "t4", "t5"];
meanError = mean(abs(results{:, "classical_" + phases} - results{:, "annotated_" + phases}), 2);
candidates = find(~isnan(meanError));
[~, pick] = min(abs(meanError(candidates) - median(meanError(candidates))));
row = results(candidates(pick), :);
trace = energy(energy.video == row.video, :);

frameTable = classical.readFrameTable();
labels = frameTable(frameTable.video == row.video, :);
[frames, hours] = classical.loadVideo(row.video, frameTable);

fig = figure("Visible", "off", "Color", "w", "Position", [100 100 1400 780]);
layout = tiledlayout(fig, 2, 4, "TileSpacing", "compact", "Padding", "compact");
title(layout, "How the classical pipeline times divisions (video " + row.video + ")", ...
    "FontSize", 14, "Color", c.ink);

ax = nexttile(layout, [1 4]);
hold(ax, "on");
energyLine = plot(ax, trace.hours, trace.energy, "Color", c.muted, "LineWidth", 1);
truth = row{1, "annotated_" + phases};
lineLabels = phases;
for k = numel(phases):-1:2
    if truth(k) - truth(k - 1) < 2  % lines this close share one label
        lineLabels(k - 1) = lineLabels(k - 1) + ", " + lineLabels(k);
        lineLabels(k) = "";
    end
end
for k = 1:numel(phases)
    truthLine = xline(ax, truth(k), "--", lineLabels(k), "Color", c.ink, ...
        "LineWidth", 1, "LabelOrientation", "horizontal", "FontSize", 11);
end
detected = row{1, "classical_" + phases};
[~, nearest] = min(abs(trace.hours - detected), [], 1);
detectedDots = plot(ax, detected, trace.energy(nearest), "o", "MarkerSize", 9, ...
    "MarkerFaceColor", c.classical, "MarkerEdgeColor", "w");
xlim(ax, [0, min(max(trace.hours), max(detected) + 15)]);
xlabel(ax, "hours after fertilization");
ylabel(ax, "motion energy (noise units)");
legend(ax, [energyLine, truthLine, detectedDots], ["motion energy inside the embryo", ...
    "embryologist's division times", "classical detection"], "Location", "northwest", "Box", "off");
styleAxes(ax);

% One frame from each stage: halfway to the next division, at most 45 minutes in
r = 1.3 * row.radius;
box = [row.center_x - r, row.center_y - r, 2 * r, 2 * r];
next = [truth(2:end), Inf];
for k = 1:numel(phases)
    ax = nexttile(layout);
    [~, f] = min(abs(hours - (truth(k) + min(0.75, (next(k) - truth(k)) / 2))));
    imshow(imcrop(frames(:, :, f), box), "Parent", ax);
    title(ax, sprintf("%d cells, %.1f h", labels.phase(f) - 1, hours(f)), ...
        "FontWeight", "normal", "FontSize", 12, "Color", c.ink);
end
exportgraphics(fig, file, "Resolution", 150);
close(fig);
end

function plotComparison(score, nVideos, file)
%PLOTCOMPARISON Share of divisions timed within 1 h, classical vs deep.
c = palette();
fig = figure("Visible", "off", "Color", "w", "Position", [100 100 900 520]);
ax = axes(fig);
stages = ["2 cells", "3 cells", "4 cells", "5 cells"] + " (" + score.phase' + ")";
bars = bar(ax, categorical(stages, stages), ...
    100 * [score.classical_within_1h, score.deep_within_1h], 0.8, "EdgeColor", "none");
bars(1).FaceColor = c.classical;
bars(2).FaceColor = c.deep;
for b = bars
    text(ax, b.XEndPoints, b.YEndPoints + 1.5, compose("%.0f%%", b.YData), ...
        "HorizontalAlignment", "center", "VerticalAlignment", "bottom", "FontSize", 11, "Color", c.ink);
end
ylim(ax, [0, 110]);
yticks(ax, 0:20:100);
ylabel(ax, "videos timed within 1 h of the embryologist (%)");
legend(ax, ["classical image processing (MATLAB)", "deep model (ResNet-34)"], ...
    "Location", "northeast", "Box", "off");
title(ax, sprintf("Division timing on %d held-out videos", nVideos), "FontWeight", "normal");
styleAxes(ax);
ax.XGrid = "off";
exportgraphics(fig, file, "Resolution", 150);
close(fig);
end

function styleAxes(ax)
%STYLEAXES Quiet axis styling shared by the report figures.
c = palette();
set(ax, "FontSize", 11, "TickDir", "out", "Box", "off", "LineWidth", 0.75, ...
    "XColor", c.muted, "YColor", c.muted, "TitleHorizontalAlignment", "left");
ax.Title.Color = c.ink;
ax.XLabel.Color = c.ink;
ax.YLabel.Color = c.ink;
grid(ax, "on");
ax.GridAlpha = 0.12;
end

function c = palette()
%PALETTE The web app's colours, used by every report figure (see
%   embryoscan/common/figures.py): amber for the classical pipeline, cobalt
%   for the deep model, ink for the embryologist and text.
c.classical = [232, 137, 12] / 255;
c.deep = [47, 91, 255] / 255;
c.ink = [13, 20, 36] / 255;
c.muted = [138, 147, 163] / 255;
end
