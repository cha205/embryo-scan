function t = detectDivisions(energy, hours)
%DETECTDIVISIONS Estimate t2, t3, t4 and t5 from motion-energy spikes.
%   t = classical.detectDivisions(energy, hours) returns a struct with the
%   hour of each division (NaN when no matching spike is found). The rules
%   encode known embryology instead of anything learned from data:
%
%     t2  the strongest spike between 18 and 38 h (the first cleavage)
%     t3  the first spike 5 to 18 h after t2
%     t4  the next spike within 4 h of t3; none means both cells divided
%         together, so t4 = t3
%     t5  the first spike 6 to 20 h after t4
%
%   Spikes are local maxima of the 3-frame moving average that stand out by
%   at least 1.5 noise units and are at least 0.5 h apart. (The separation
%   is converted to frames because the video timestamps are rounded to
%   0.1 h, so neighbouring frames can share one.)

arguments
    energy (:, 1) double
    hours (:, 1) double
end

smoothed = movmean(energy, 3);
frameStep = max(median(diff(hours)), eps);
isSpike = islocalmax(smoothed, "MinProminence", 1.5, ...
    "MinSeparation", max(1, round(0.5 / frameStep)));
spikeHours = hours(isSpike);
spikeSize = smoothed(isSpike);

t = struct("t2", NaN, "t3", NaN, "t4", NaN, "t5", NaN);

window = find(spikeHours >= 18 & spikeHours <= 38);
if isempty(window)
    return
end
[~, k] = max(spikeSize(window));
t.t2 = spikeHours(window(k));

t.t3 = firstAfter(spikeHours, t.t2 + 5, t.t2 + 18);
if isnan(t.t3)
    return
end
t.t4 = firstAfter(spikeHours(spikeHours > t.t3), t.t3, t.t3 + 4);
if isnan(t.t4)
    t.t4 = t.t3;
end
t.t5 = firstAfter(spikeHours, t.t4 + 6, t.t4 + 20);
end

function h = firstAfter(spikeHours, from, to)
%FIRSTAFTER First spike time in the window [from, to], or NaN.
inWindow = spikeHours(spikeHours >= from & spikeHours <= to);
if isempty(inWindow)
    h = NaN;
else
    h = min(inWindow);
end
end
