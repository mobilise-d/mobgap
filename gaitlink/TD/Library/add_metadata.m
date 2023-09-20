function add_metadata(dirname, filename, textRow)
    fid = fopen(fullfile(dirname, filename), 'a');
    fprintf(fid, strcat(textRow, '\n'));
    fclose(fid);
end
