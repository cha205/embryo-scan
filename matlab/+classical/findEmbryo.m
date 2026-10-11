function [center, radius, method] = findEmbryo(img, options)
%FINDEMBRYO Locate the embryo in a frame.
%   [center, radius, method] = classical.findEmbryo(img) returns the embryo
%   centre [x y] in pixels, its radius and how it was found:
%
%     "hough"    a circular Hough transform (imfindcircles) found its outline.
%                The well's bright rim is a much larger circle, so the radius
%                range leaves it out.
%     "texture"  no circle: the centre of the most textured region
%                (classical.textureCenter), radius NaN
%     "none"     nothing textured either: the image centre, radius NaN
%
%   Embryos often rest against the side of the well, far from the image
%   centre, so circles are scored by strength alone.
%
%   Options:
%     RadiusRange  [min max] radius in pixels (default [35 68], for frames
%                  scaled to 250 x 250)
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
