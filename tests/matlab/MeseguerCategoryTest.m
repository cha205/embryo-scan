classdef MeseguerCategoryTest < matlab.unittest.TestCase
    % Tests for classical.meseguerCategory, which mirrors meseguer_category()
    % in embryoscan/timelapse/labels.py.

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function onTimeAndInSync(testCase)
            % t5 inside 48.8-56.6 h, s2 = 0.5 h, cc2 = 11 h
            t = struct("t2", 26, "t3", 37, "t4", 37.5, "t5", 50);
            testCase.verifyEqual(classical.meseguerCategory(t), "A+");
        end

        function lateOutOfSyncAndSlow(testCase)
            % t5 outside the window, s2 = 2 h, cc2 = 13 h
            t = struct("t2", 26, "t3", 39, "t4", 41, "t5", 60);
            testCase.verifyEqual(classical.meseguerCategory(t), "D-");
        end

        function neverReachingFiveCellsIsOutsideTheWindow(testCase)
            t = struct("t2", 26, "t3", 37, "t4", 37.5, "t5", NaN);
            testCase.verifyEqual(classical.meseguerCategory(t), "C+");
        end

        function cannotScoreWithoutFourCells(testCase)
            t = struct("t2", 26, "t3", 37, "t4", NaN, "t5", NaN);
            testCase.verifyEqual(classical.meseguerCategory(t), "X");
        end
    end
end
