classdef FindEmbryoTest < matlab.unittest.TestCase
    % Tests for classical.findEmbryo on synthetic frames with a known answer.

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function findsADrawnEmbryo(testCase)
            rng(1);
            [xx, yy] = meshgrid(1:250, 1:250);
            r = hypot(xx - 120, yy - 132);
            img = 0.55 + 0.01 * randn(250);                      % smooth culture medium
            inside = r < 50;
            img(inside) = 0.45 + 0.08 * rand(nnz(inside), 1);  % grainy cells
            img(abs(r - 50) < 2) = 0.2;                          % dark membrane edge
            [center, radius, method] = classical.findEmbryo(img);
            testCase.verifyEqual(method, "hough");
            testCase.verifyLessThan(norm(center - [120 132]), 3);
            testCase.verifyLessThan(abs(radius - 50), 4);
        end

        function fallsBackToTextureWithoutAnOutline(testCase)
            % grainy cells with no circular edge, off to one side of the well
            rng(5);
            [xx, yy] = meshgrid(1:250, 1:250);
            img = 0.55 + 0.005 * randn(250);
            patch = hypot(xx - 80, yy - 150) < 45;
            img(patch) = img(patch) + 0.15 * randn(nnz(patch), 1) .* exp(-hypot(xx(patch) - 80, yy(patch) - 150) / 20);
            [center, radius, method] = classical.findEmbryo(img);
            testCase.verifyEqual(method, "texture");
            testCase.verifyLessThan(norm(center - [80 150]), 8);
            testCase.verifyTrue(isnan(radius));
        end

        function returnsImageCentreWhenNothingIsThere(testCase)
            [center, radius, method] = classical.findEmbryo(0.5 * ones(250));
            testCase.verifyEqual(center, [125 125]);
            testCase.verifyTrue(isnan(radius));
            testCase.verifyEqual(method, "none");
        end
    end
end
