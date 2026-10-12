function energy = motionEnergy(frames, center, radius)
%MOTIONENERGY Frame-to-frame change inside the embryo.
%   energy = classical.motionEnergy(frames, center, radius) returns, for each
%   frame, the mean absolute difference from the previous frame inside the
%   embryo after light smoothing and brightness matching, as a robust z-score
%   against the video's typical change. Divisions show up as spikes because a
%   cell splitting in two changes many pixels at once.

arguments
    frames (:, :, :) uint8
    center (1, 2) double
    radius (1, 1) double {mustBePositive}
end

[h, w, n] = size(frames);
[xx, yy] = meshgrid(1:w, 1:h);
inside = (xx - center(1)).^2 + (yy - center(2)).^2 <= (0.95 * radius)^2;

energy = zeros(n, 1);
previous = [];
for t = 1:n
    f = imgaussfilt(im2double(frames(:, :, t)), 1.5);
    f = f / mean(f(inside));  % cancel frame-wide brightness flicker
    if ~isempty(previous)
        d = abs(f - previous);
        energy(t) = mean(d(inside));
    end
    previous = f;
end
if n > 1
    energy(1) = energy(2);
end

typical = median(energy);
spread = 1.4826 * median(abs(energy - typical)) + eps;
energy = (energy - typical) / spread;
end
