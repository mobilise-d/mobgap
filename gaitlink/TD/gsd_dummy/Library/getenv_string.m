%getenv_number   Reads the value of an environment variable and evals
%                it as a string. If the variable does not exist, then
%                the default_value is returned
%
%   function [n]=getenv_string(name, default_value)
%
%   Inputs:
%    - name: The name of the environment variable to search for
%    - default_value: The string that will be returned if the environment
%      + variable does not exist
%
%   Output:
%    - The string(raw) value of the environment variable
%
function [s]=getenv_string(name, default_value)
    value = getenv(name);
    if isempty(value)==0
        s =value;
        return;
    else
        s = default_value;
        return;
    end

