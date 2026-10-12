function [center, radius, method] = findEmbryo(img, options)
%FINDEMBRYO Locate the embryo in a frame.
%   [center, radius, method] = classical.findEmbryo(img) returns the embryo
%   center [x y] and radius in pixels, and how it was found:
%
%     "hough"    circular Hough transform; the radius range excludes the much
%                larger rim of the well
%     "texture"  no circle, so the most textured region is used
%                (classical.textureCenter) and radius is NaN
%     "none"     no texture either: the image center, radius NaN
%
%   Circles are ranked by strength only, since embryos often rest against
%   the side of the well, far from the image center.
%
%   Options:
%     RadiusRange  [min max] radius in pixels (default [35 68] at 250 x 250)
%     Sensitivity  imfindcircles sensitivity (default 0.93)

arguments
    img {mustBeNumeric}
    options.RadiusRange (1,2) double {mustBePositive} = [35 68]
    options.Sensitivity (1,1) double = 0.93
end

img = im2double(img);
[centers, radii] = imfindcircles(img, options.RadiusRange, ...
    "Sensitivity", options.Sensitivity, "Method", "TwoStage");
if ~isempty(centers)
    center = centers(1, :);  % sorted strongest first
    radius = radii(1);
    method = "hough";
    return
end

radius = NaN;
center = classical.textureCenter(img, Smoothing=options.RadiusRange(1) / 3);
method = "texture";
if any(isnan(center))
    center = size(img, [2 1]) / 2;
    method = "none";
end
end
