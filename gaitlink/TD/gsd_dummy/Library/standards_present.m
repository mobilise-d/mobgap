function valid = standards_present(data)
if isfield(data, 'Standards')
    valid = 1;
else
    valid = 0;
end
end

