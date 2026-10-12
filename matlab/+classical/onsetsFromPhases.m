function on = onsetsFromPhases(phase, hours)
%ONSETSFROMPHASES Hour each early division was reached, from per-frame phases.
%   on = classical.onsetsFromPhases(phase, hours) returns a struct with t2,
%   t3, t4 and t5: the first hour at which the phase was that division's or
%   later, so going from 2 cells straight to 4 gives t3 = t4. Divisions never
%   reached are NaN. Mirrors onsets() in embryoscan/timelapse/labels.py and
%   uses its PHASES indices (t2 is 3, -1 is unlabelled).

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
