function center = textureCenter(img, options)
%TEXTURECENTER Center of the most textured region of a frame.
%   center = classical.textureCenter(img) returns [x y] of the peak of
%   smoothed local contrast (stdfilt). Cells are grainy and the culture medium
%   is smooth, so the peak lands on the embryo even when its outline is too
%   faint for a Hough transform. Very bright pixels (the well's rim) and a
%   border are ignored. Returns NaN for a featureless image.
%
%   Options:
%     Smoothing  Gaussian sigma in pixels (default 12 at 250 x 250)

arguments
    img {mustBeNumeric}
    options.Smoothing (1, 1) double {mustBePositive} = 12
end

img = im2double(img);
texture = stdfilt(img, true(5));
texture(img > 0.8) = 0;
texture = imgaussfilt(texture, options.Smoothing);

border = round(0.15 * size(img, [1 2]));
texture([1:border(1), end - border(1) + 1:end], :) = 0;
texture(:, [1:border(2), end - border(2) + 1:end]) = 0;

[peak, k] = max(texture(:));
if peak < 1e-3
    center = [NaN NaN];
    return
end
[y, x] = ind2sub(size(texture), k);
center = [x y];
end
