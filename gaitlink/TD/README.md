# Turning detection
This project contains the code of the turning detection algorithm decribed in El-Gohary et al.  2014.

### Inputs
Run the driver.m file and specifcy folder paths 
- indir, where data will be looked for as input: The `SD` block requires the prepared `data.mat` file and also the output of a previously performed `GSD` or `GSD_Dummy` block, called `GSD_Output.mat`.
- outdir, where the output files will be stored.

To select a specific sensor type, specify the `sensor_string` in `driver.m` according to the selection listed in `./WorkflowBlock/src/main/resources/service.xml`.

### Outputs

Four output files will be stored in the `outdir` in  `.json` and `.mat` format, respectively, and both with an algorithm generic and algorithm specific file name.
The `Turning_ElGohary` algorithm provides as output the regions of turnings with start and end in seconds since the start of the recording / trial, and several related parameters like the turning angle or the peak angular velocity.

## References
El-Gohary, M., Pearson, S., McNames, J., Mancini, M., Horak, F., Mellone, S., & Chiari, L. (2014). Continuous monitoring of turning in patients with movement disability. Sensors (Basel, Switzerland), 14(1), 356–369. https://doi.org/10.3390/s140100356
