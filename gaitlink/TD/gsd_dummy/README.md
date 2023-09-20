# GSD_Dummy

The `GSD_Dummy` can be used to extract the gait sequences as defined by a reference system and write them into an output file that look identical as the ones created by the "real" `GSD` algorithms.
This can be useful when one of the subsequent blocks in the pipeline should be tested in isolation, where the influence or bias of the actual `GSD` algorithm should be avoided / circumvented.

### Inputs
Run the driver.m file and specifcy folder paths 
- indir, where data will be looked for as input: The `GSD_Dummy` block only requires the prepared `data.mat` file.
- outdir, where the output files will be stored.

To select a specific gold standard and gait bout type, specify the `standard_string` and the `bout_string` in `driver.m` according to the selection listed in `./WorkflowBlock/src/main/resources/service.xml`.

### Outputs

Four output files will be stored in the `outdir` in  `.json` and `.mat` format, respectively, and both with an algorithm generic and algorithm specific file name.
The output contains a structure similar to the data.mat file and provides the “Start” and “End” of gait sequences in seconds relative to the start of the recording / the test trial.
