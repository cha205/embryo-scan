classdef OnsetsFromPhasesTest < matlab.unittest.TestCase
    % Tests for classical.onsetsFromPhases, which mirrors onsets() in
    % embryoscan/timelapse/labels.py (phase index 3 is t2, -1 is unlabelled).

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function skippedPhaseCountsAsReached(testCase)
            % unlabelled, 1 cell, 2 cells, straight to 4 cells, then 5 cells
            phase = [-1 -1 2 2 3 3 5 5 6]';
            on = classical.onsetsFromPhases(phase, (1:9)');
            testCase.verifyEqual([on.t2 on.t3 on.t4 on.t5], [5 7 7 9]);
        end

        function unreachedDivisionIsNaN(testCase)
            on = classical.onsetsFromPhases([2 3 3 4]', (1:4)');
            testCase.verifyEqual([on.t2 on.t3], [2 4]);
            testCase.verifyTrue(isnan(on.t4) && isnan(on.t5));
        end
    end
end
