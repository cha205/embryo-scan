classdef DetectDivisionsTest < matlab.unittest.TestCase
    % Tests for classical.detectDivisions on synthetic motion-energy traces.

    properties
        Hours = (0:0.25:80)'  % one frame every 15 minutes, like the videos
    end

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function recoversKnownDivisionTimes(testCase)
            rng(2);
            energy = 0.3 * randn(size(testCase.Hours)) + testCase.spikesAt([26 37 38 50]);
            t = classical.detectDivisions(energy, testCase.Hours);
            testCase.verifyEqual([t.t2 t.t3 t.t4 t.t5], [26 37 38 50], "AbsTol", 0.5);
        end

        function synchronousDivisionGivesEqualT3AndT4(testCase)
            energy = testCase.spikesAt([26 38 50]);
            t = classical.detectDivisions(energy, testCase.Hours);
            testCase.verifyEqual(t.t3, t.t4);
            testCase.verifyEqual(t.t5, 50, "AbsTol", 0.5);
        end

        function flatTraceFindsNothing(testCase)
            t = classical.detectDivisions(zeros(size(testCase.Hours)), testCase.Hours);
            testCase.verifyTrue(isnan(t.t2) && isnan(t.t5));
        end
    end

    methods
        function energy = spikesAt(testCase, spikeHours)
            energy = zeros(size(testCase.Hours));
            for h = spikeHours
                energy = energy + 8 * exp(-((testCase.Hours - h) / 0.3).^2);
            end
        end
    end
end
