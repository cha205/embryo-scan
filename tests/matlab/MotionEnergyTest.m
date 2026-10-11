classdef MotionEnergyTest < matlab.unittest.TestCase
    % Tests for classical.motionEnergy on a synthetic image stack.

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function aSuddenChangeIsTheBiggestSpike(testCase)
            rng(3);
            before = uint8(100 + 40 * rand(60));
            after = uint8(100 + 40 * rand(60));  % the "cells" rearrange at frame 6
            frames = cat(3, repmat(before, 1, 1, 5), repmat(after, 1, 1, 5));
            frames = frames + uint8(randi([0 2], size(frames)));  % camera noise
            energy = classical.motionEnergy(frames, [30 30], 25);
            [~, peak] = max(energy);
            testCase.verifyEqual(peak, 6);
        end

        function stillVideoHasNoLargeSpike(testCase)
            rng(4);
            frames = repmat(uint8(100 + 40 * rand(60)), 1, 1, 10) + uint8(randi([0 2], 60, 60, 10));
            energy = classical.motionEnergy(frames, [30 30], 25);
            testCase.verifyLessThan(max(energy), 6);
        end
    end
end
