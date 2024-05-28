# Common Datatypes


When considering data in mobgap, we need to separate two hierarchies:

1. The direct input and outputs of a single algorithm. There we try to stick to common datatypes as much as possible.
2. Full datasets that are used as inputs for pipelines. 
   These are based on {external:class}`tpcp.Dataset` and don't just contain all the data required for a single recording, 
   but also additional metadata, the logic on how to load the data, and usually the structure of an entire set of 
   recordings that together form a dataset.

We will look at both of these in the following sections.

## Common low level datatypes for algorithms

Algorithms only require the data as input that they really need.
This should make it possible to use algorithms easily across different datasets.

To make it possible for you to easily create these input datatypes from your own data, we stick to very basic containers
with only a few rules associated with them.
Most structure are {external:class}`pandas.DataFrame` and only the expected columns and index shapes change between 
different algorithm inputs/outputs.

Below we will list the datatypes in the order that they typically appear in a gait pipeline.
But before some general rules that apply to all datatypes:

### Units

Some common units:

| Value           | Unit                               | Short-hand Postfix |
|-----------------|:-----------------------------------|:-------------------|
| Acceleration    | m/s^2                              | -                  |
| Rotation Rate   | deg/s                              | -                  |
| Magnetic fields | uT                                 | -                  |
| Velocity        | m/s                                | _mps               |
| Distance        | m                                  | _m                 |
| Cadence         | steps/min                          | _spm               |         
| Time/Duration   | s or # (see "Further Rules" below) | _s/_samples        |         
| Sampling Rate   | Hz                                 | _hz                |

Further rules:

- In all (but the most common cases), the unit is also appended to the column/variable name.
  This is to make it clear what the unit of the value is and to avoid confusion.
- If a method requires a parameter in a given unit, append it with a common short-hand name in this unit (e.g.
  `windowsize_ms` would expect a value in milliseconds).
- Time is either specified in seconds (s) for user facing durations (e.g. stride time), but time points in intermediate
  results (e.g. biomechanical events) are typically specified in samples since the start of the measurement (#).

### Start-End Indices

Many of the datatypes contain information about the start or the end of a certain time-period (e.g. a gait sequence).
Start and end values are (whenever possible) provided in samples from the start of a recording.
The respective time-period is then defined as [start, end), meaning starting with the start sample (inclusive) until the
end sample (exclusive).
This follows the Python convention for indices and therefore, you can extract a region from the dataset as follows:

```python-console
>>> dataset.iloc[start : end]
```

Note that `dataset.iloc[end]` refers to the first value **after** the region!
To get the last sample of a region you must use `end-1`.

The duration of a region (in samples) is simply `end - start`.

For edge cases this means:

- A region that starts on the first sample of a dataset has `start=0`
- A region that ends with the dataset (i.e. inclusive the last sample) has `end=len(dataset)`.
- If two regions are directly adjacent to each other, the end index of the first, is the start index of the second.


### Raw IMU Data

Raw IMU data is a simple dataframe with the following columns:

- `acc_{x,y,z}`: The acceleration in x, y, and z direction in **m/s^2**
- `gyr_{x,y,z}`: The angular velocity in x, y, and z direction in **deg/s**
- `mag_{x,y,z}`: The magnetic field in x, y, and z direction in **uT**

The data does not need to contain all 3 sensor types, but this will limit the algorithms that can be applied to it.

We don't require a specific index for the data, as all processing internally will be performed on under the assumption
that the data represents equally spaced samples with a fixed sampling rate.

### Gait Sequences and Walking Bouts

Gait sequences and walking bouts simply represent regions within the data.
Hence, they are represented as a dataframe with the following mandatory columns:

- `start`: The start of the sequence in samples from the start of the data
- `end`: The end of the sequence in samples from the start of the data

The start and end values follow the typical Python indexing rules (see above).

We don't require any specific index for the data, but we suggest using a unique identifier for each sequence.
In most cases internally, we use a simple integer index starting at 1.
So the first WB/GS in the data would have the index 1, the second 2, and so on.

Depending on the type of sequence (i.e. WB or GS) we name the index column either `gs_id` or `wb_id`.
However, this does not affect the processing of the data.

For more information on GS and WB, see the [Q&A](#q&a__wb_vs_gs).

### Gait Events

For gait events, we use a simple dataframe with usually just a single column referring to the sample index of the event.
For example for initial contacts, we would have a column `ic` that contains the sample index of the initial contacts 
relative to the start of the data.

Note, that like with the gait sequences, the context (i.e. what piece of data the events were extracted from) matters.
As all event detection algorithms are expected to work on individual gait sequences, the sample indices that you 
encounter in algorithm outputs are usually relative to the start of the gait sequence.
So if you need them relative to the start of the recording you need to adjust the values accordingly.
Similar, with reference data we usually differentiate between `reference_parameters_` and 
`reference_parameters_relative_to_wb_`.
For more information see {py:class}`~mobgap.data.base.BaseGaitDatasetWithReference`.

Like with the other structures, we don't have strict assumptions on the index.
However, we suggest using a unique identifier for each event starting with 1 for the first event.
Internally, we name the index column `step_id`, as relevant event usually occur once per step.
But this naming does not affect the processing of the data.

This event dataframe can have additional columns that contain additional information about the events.
For example, the `ic` column is often accompanied by a `lr_label` column that contains the label of the leg 
(`left`/`right`) that the initial contact was detected on.
For more information see {py:mod}`~mobgap.lrd`.

### Per-Second Gait Parameters

For the Cadence and Stride Length estimation (and potentially other parameters in the future) we expect the output to be
on a per-second level (i.e. one parameter per second of data).
Learn more about the reasoning behind this in the [Q&A](#q&a__sec_vals).
These dataframes have one or multiple columns that contain the respective parameter (e.g. `cad_spm`).
The name of the column should be the parameter name or a common abbreviation of it joined with the unit of the parameter
(e.g. `cad_spm` for cadence in steps per minute).

The index of the dataframe contains the sample marking the center of the respective second relative to the start of the
data used to calculate the parameter and usually has the name `sec_center_samples`.

Having the actual sample in the index (instead of just numbering the seconds) allows us to easily align the per-second
parameters with the original data or adjust the data offset if needed.

### Aggregated Parameters

TODO


### Multi-index Dataframes and their usecases

In some cases, we use the dataframes described above, but with a multi-index.
This usually happens when we want to represent a group of results/data in one dataframe.
The most common usecase of this is in the output of the {py:class}`~mobgap.pipeline.GsIterator` class.
It allows to apply a set of algorithms to each gait sequence in a dataset and then combine the results in a single
dataframe.

These output dataframes then have a multi-index with the first level being the `gs_id` and the second level being the
original index of the respective algorithm output.
Algorithms and blocks designed to work with these outputs (like the {py:class}`~mobgap.wba.WbAssembly`) are designed
to handle these multi-index dataframes.

Further, we use multi-index dataframes to represent aggregated parameters of multiple recordings/participants/etc.

## Datasets

Compared to the low level datatypes, datasets are higher level abstractions, containing all data and metadata 
associated with a set of recordings.
They are based on the {external:class}`tpcp.Dataset` class and allow to easily load and access otherwise complex data 
structures.

A dataset that has only one "row" (i.e. one recording) is referred to as a "datapoint" and is the expected input for
all the pipelines in mobgap.

Using this dataset abstraction allows us to easily apply the same algorithms to different datasets and to use 
higher-level tpcp-features like the {py:class}`~tpcp.validate.cross_validate` to run and evaluate our pipelines on
subsets of our datasets in a consistent manner.

The simplest dataset that we provide out of the box is the {py:class}`~mobgap.data.LabExampleDataset`, which can be
used to load the example data that we provide with mobgap.

If you have already loaded your own data and want to use it with a mobgap pipeline, you can use the 
{py:class}`~gaitlin.data.GaitDatasetFromData` class to quickly create a compatible dataset from your data.
However, we highly encourage you to create a custom dataset class for your data.
This will simplify a lot of things and provides a generally nice abstraction for your dataset.

We provide helper for creating custom datasets in the {py:mod}`~mobgap.data` module and are happy to integrate
developed datasets of public datasets into mobgap.

TODO: Link to tutorial on working with data

