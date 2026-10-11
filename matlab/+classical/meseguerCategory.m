function category = meseguerCategory(t)
%MESEGUERCATEGORY Meseguer 2011 embryo category from division times.
%   category = classical.meseguerCategory(t) takes a struct with t2, t3, t4
%   and t5 (hours after fertilization) and returns "A+" ... "D-", or "X"
%   when it can't be scored. Mirrors meseguer_category() in
%   embryoscan/timelapse/labels.py:
%
%     t5 inside 48.8-56.6 h         A or B, otherwise C or D
%     s2 = t4 - t3 at most 0.76 h   A or C, otherwise B or D
%     cc2 = t3 - t2 at most 11.9 h  "+", otherwise "-"

arguments
    t (1, 1) struct
end

s2 = t.t4 - t.t3;
cc2 = t.t3 - t.t2;
if isnan(s2) || isnan(cc2)
    category = "X";
    return
end

inWindow = t.t5 >= 48.8 && t.t5 <= 56.6;  % false when t5 is NaN
inSync = s2 <= 0.76;
if inWindow && inSync
    letter = "A";
elseif inWindow
    letter = "B";
elseif inSync
    letter = "C";
else
    letter = "D";
end
if cc2 <= 11.9
    category = letter + "+";
else
    category = letter + "-";
end
end
