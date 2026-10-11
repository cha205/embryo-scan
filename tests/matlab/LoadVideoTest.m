classdef LoadVideoTest < matlab.unittest.TestCase
    % Tests for classical.loadVideo on a small fake video written to a
    % temporary data folder.

    methods (TestClassSetup)
        function addCodeToPath(testCase)
            root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
            testCase.applyFixture(matlab.unittest.fixtures.PathFixture(fullfile(root, "matlab")));
        end
    end

    methods (Test)
        function missingFrameRepeatsThePreviousOneAtTheSameSize(testCase)
            import matlab.unittest.fixtures.EnvironmentVariableFixture
            import matlab.unittest.fixtures.TemporaryFolderFixture
            data = testCase.applyFixture(TemporaryFolderFixture).Folder;
            testCase.applyFixture(EnvironmentVariableFixture("EMBRYOSCAN_DATA", data));
            folder = fullfile(data, "raw", "gomez2022", "embryo_dataset", "AB1-1");
            mkdir(folder);
            for frame = [1 2 4]  % frame 3 is missing
                imwrite(uint8(60 * frame * ones(40)), fullfile(folder, sprintf("D2014_WELL1_RUN%d.jpeg", frame)));
            end
            % frame 0 has no image, as in some of the dataset's timestamp files
            frameTable = table(repmat("AB1-1", 5, 1), (0:4)', [0.75; 1; 1.25; 1.5; 1.75], ...
                'VariableNames', ["video", "frame", "hours"]);

            [frames, hours] = classical.loadVideo("AB1-1", frameTable, Scale=0.5);

            testCase.verifySize(frames, [20 20 5]);
            testCase.verifyEqual(frames(:, :, 1), frames(:, :, 2));
            testCase.verifyEqual(frames(:, :, 4), frames(:, :, 3));
            testCase.verifyGreaterThan(mean(frames(:, :, 5), "all"), mean(frames(:, :, 3), "all"));
            testCase.verifyEqual(hours, [0.75; 1; 1.25; 1.5; 1.75]);
        end
    end
end
