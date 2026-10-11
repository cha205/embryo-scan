function on = onsetsFromPhases(phase, hours)
%ONSETSFROMPHASES Hour each early division was reached, from per-frame phases.
%   on = classical.onsetsFromPhases(phase, hours) returns a struct with
%   fields t2, t3, t4 and t5: the first hour at which the frame's phase was
%   that division's or later. Skipped phases count as reached, so an embryo
%   going from 2 cells straight to 4 gets t3 = t4. Divisions never reached
%   are NaN. Mirrors onsets() in embryoscan/timelapse/labels.py; phase
%   indices follow its PHASES list (t2 is 3) and -1 marks unlabelled frames.

arguments
    phase (:, 1) double
    hours (:, 1) double
end

names = ["t2", "t3", "t4", "t5"];
firstPhase = [3, 4, 5, 6];
for k = 1:numel(names)
    reached = find(phase >= firstPhase(k), 1);
    if isempty(reached)
        on.(names(k)) = NaN;
    else
        on.(names(k)) = hours(reached);
    end
end
end
