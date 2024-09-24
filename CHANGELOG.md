# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2024-09-13 

### Scientific Changes

- **BREAKING**: Using "high precision" standard value of gravity everywhere (9.80665 m/s^2). 
  Before we were using 9.81 m/s^2 during the loading of MobiliseD matlab files and when calculating the stride length 
  validity threshold.
  This should have a small impact on the results of most algorithms, hence, we marked it as a breaking change.

### Added

- The `apply_aggregations` and `apply_transformation` functions now have an option (on by default) to ignore 
  transformations that expect columns that don't exist.
  This should allow the use of the default aggregations and transformations in more situations.

### Changed

- The matlab loader now accepts infoForAlgo files that are missing some of the optional fields.


## [0.7.0] - 2024-09-13

### Scientific Changes

- **CRITICAL**: The default thresholds for both variants of the GSDIonescu algorithms were not correctly adapted to the
  input data being in m/s2 instead of g.
  With the old threshold far to many gait sequences were detected.
  This fix should have a substantial positive impact on the performance of the Impaired pipeline.

### Fixed
- `as_samples` now correctly preserves the index of a dataframe.

## [0.6.0] - 2024-07-26

### Added

- Evaluation Examples for CAD and Stride Length (https://github.com/mobilise-d/mobgap/pull/174)

### Changed

- The reference data parser now allows missing information. In case either no turns, no ICs or no strides are available,
  the respective attribute of the reference data is set to None. Before this threw an error.

## [0.5.1] - 2024-07-24

- Correctly specified minimal pandas version as >=2.2.0

## [0.5.0] - 2024-07-16

### Changed

- **IMPORTANT**:All algorithms now require the input to be in the body frame.
  This ensures that the correct axis is used in all cases.
  (https://github.com/mobilise-d/mobgap/pull/170, https://github.com/mobilise-d/mobgap/pull/171)
- The Coordinate Sytem guide was reworked to cover all different coordinate systems and hopefully prevent common 
   mistakes.
  (https://github.com/mobilise-d/mobgap/pull/170)
- `BaseMobilisedPipeline` is renamed to `GenericMobilisedPipeline`. `BaseMobilisedPipeline` is now the "actual" 
  baseclass for Mobilise-D pipelines, without any implementation.
- `MobilisedMetaPipeline` has been renamed to `MobilisedPipelineUniversal`.
  This should make the naming more consistent (compare `MobilisedPipelineHealthy`, `MobilisedPipelineImpaired`).
- `MobilisedPipelineUniversal` now has proxy attributes for all the primary results expected in `BaseMobilisedPipeline` and
  can be used as a drop-in replacement for `GenericMobilisedPipeline`.
- Gaitmap has been removed as a dependency.
  All used functions are now vendored.
  This should reduce the number of dependency mobgap pulls in.
  (https://github.com/mobilise-d/mobgap/pull/166)
- Significant performance improvements for GsdAdaptiveIonescu.
  (https://github.com/mobilise-d/mobgap/pull/167). Thanks @tbonewmy !

### Fixed

- **IMPORTANT**: The TD Elgohary algorithm was using the wrong axis for the calculation of the angle.
  This was fixed and the results should now be correct.
  (https://github.com/mobilise-d/mobgap/pull/170)
- The turning and stride length algorithm both had an option to be used with prior orientation estimation.
  However, this did not work due to a coordinate system mismatch.
  The Madgwick algorithm has been modified to allow returning data in the correct coordinate system.
  (https://github.com/mobilise-d/mobgap/pull/170)
- In case no index-col-names are passed to the `DatasetFromData` class and the data index is just a string and not a 
  tuple, a generic column (`level_0`) is correctly created. (https://github.com/mobilise-d/mobgap/issues/164)
- The coordinate system was not documented correctly. y and z axis were flipped in the description.

### Added

- The Madgwick algorithm is now available in the `mobgap` package.
  The implementation is copied from gaitmap.
  Note, that its implementation differs slightly from the original Madgwick implementation and it supports the concept
  of body frame data, that only exists in mobgap.
  (https://github.com/mobilise-d/mobgap/pull/166, https://github.com/mobilise-d/mobgap/pull/170)
- A bunch new helper tools to perform coordinate transforms.
  (https://github.com/mobilise-d/mobgap/pull/170)
- Added various array and rotation helpers that were originally imported from gaitmap.
  (https://github.com/mobilise-d/mobgap/pull/166)
- Added explicit evaluation challenges for the Gait Sequence Detection algorithms.
  (https://github.com/mobilise-d/mobgap/pull/169)

## [0.4.0] - 2024-06-17

### Added

- Example on how to create custom datasets (https://github.com/mobilise-d/mobgap/pull/155, https://github.com/mobilise-d/mobgap/pull/145)
- Pipelines to run the P1 and P2 Mobilise-D pipelines (https://github.com/mobilise-d/mobgap/pull/145)
- A "naive" walking speed calculation method that just multiplies the cadence with the step length.
  (https://github.com/mobilise-d/mobgap/pull/148)
- The Zjilstra Stride/Step length algorithm (https://github.com/mobilise-d/mobgap/pull/142)
- A new GSD algorithm called ``GsdIonescu`` (https://github.com/mobilise-d/mobgap/pull/143)
- A new GSD algorithm called ``GsdAdaptiveIonescu`` (https://github.com/mobilise-d/mobgap/pull/53)
- The ElGohary Turing Detection algorithm (https://github.com/mobilise-d/mobgap/pull/131)
- The ``iter_gs`` method now has a new argument ``id_col_name`` to specify the column of the gs_list that should be 
  used to infer the id of the returned regions.
  Even without the argument, the method will try to infer the id column.
  This allows the use with other region lists besides gs/wb lists.
  (https://github.com/mobilise-d/mobgap/pull/135)
- The ``GsIterator`` now has a way to iterate sub-regions using ``iter_subregions``, the ``subregion`` context manager
  and the ``with_subregion`` method.
  This allows for nested iteration over regions and is the basis of our support for "refined GaitSequence".
  (https://github.com/mobilise-d/mobgap/pull/135)
- A ``refine_gs`` method that returns a new gait sequence that starts from the first IC and ends at the last IC.
  This can be used with the new subregion iteration to iterate over the subregions of a gait sequence.
  (https://github.com/mobilise-d/mobgap/pull/135)
- The MobiliseDAggregator can now take `None` as grouping parameter, which results in all WBs being aggregated together.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The Multi-Df groupby now has a way to pass parameters to the underlying `.groupby` call.
  (https://github.com/mobilise-d/mobgap/pull/141)
- A method to generate a stride list from initial contacts (`strides_list_from_ic_lr_list`).
  (https://github.com/mobilise-d/mobgap/pull/141)
- A method to interpolate per-sec values to regions (usually strides) (`naive_sec_paras_to_regions`).
  (https://github.com/mobilise-d/mobgap/pull/141)
- All the loader functions for the matlab format now have the option to skip a test, if either data or reference data 
  is not available (https://github.com/mobilise-d/mobgap/pull/125)
- Matlab-loader dataclasses now have the option to use a pre-computed index instead of deriving the test list by 
  loading the Matlab file itself.
  This should help reducing the initial index creation time for large datasets.
  (https://github.com/mobilise-d/mobgap/pull/125)
- A loader for the Mobilise-D TVS dataset, which will be published end on June.
  (https://github.com/mobilise-d/mobgap/pull/125)
- General methods to calculate and aggregate error metrics for final WB-level parameters.
  (https://github.com/mobilise-d/mobgap/pull/126)


### Changed

- Renamed algorithm folders to easier to understand names(gsd -> gait_sequences, icd -> initial_contacts, lrc -> laterality)
  (https://github.com/mobilise-d/mobgap/pull/156)
- Many algorithms are now more robust to short data and correctly return empty results if the data is too short instead
  of erroring out.
  (https://github.com/mobilise-d/mobgap/pull/145)
- The Gait Sequence iterator does not allow for dynamic attribute access anymore.
  Only the use of the ``results_`` object is allowed.
  (https://github.com/mobilise-d/mobgap/pull/135)
- Aggregations for Typed iterators/Gait Sequence iterators only take a single argument now.
  This is a list of return type tuples that contain ALL results (before the agg funcs only got the values for one of 
  the result attributes) and the input, but also additional context information that can be extended by the iterator 
  object.
  (https://github.com/mobilise-d/mobgap/pull/135)
- Reference WB ids now start at 0 again instead of 1.
- Reference parameters like turns and initial contacts that exist per WB are now numbered per WB.
- The MobilseDAggregator now uses new more expressive names by default.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The expected Cadence output now has a new column name `cadence_spm` instead of `cad_spm`.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The result attribute for Cadence in all Cadence algorithms and the GSIteration is now called 
  `cadence_per_sec` instead of `cad_per_sec`.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The Mobilise-D datasets `metadata` attribute is renamed to `recording_metadata` and is now a dictionary instead of a 
  named tuple.
  It also contains more information about the recording.
  (https://github.com/mobilise-d/mobgap/pull/141)
- All file/directory based versions of the Mobilise-D datasets now require a `measurement_condition` argument.
  (https://github.com/mobilise-d/mobgap/pull/141)
- All datasets now have a `participant_metadata` attribute that contains information about the participant.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The Cadence method does now include "incomplete" seconds.
  This means the "partial" last second of a recording is now included in the output.
  This ensures that all strides are covered by the output.
  (https://github.com/mobilise-d/mobgap/pull/141)
- The MobiliseDAggregator does not round the output anymore and further does not convert stride length values to "cm"
  anymore.
  If this is desired, the user has to do this manually.
  (https://github.com/mobilise-d/mobgap/pull/145)

### Fixed

- The check that we assume a gs to start and end with an IC was not correctly performed for the end IC and a warning
  was raised incorrectly. This is now fixed.
  (https://github.com/mobilise-d/mobgap/pull/135)
- The ``GsIterator`` does not throw an error anymore, if the GS list was empty.
  (https://github.com/mobilise-d/mobgap/pull/135)
- The reference parameters for turns loaded from Matlab files now have the correct units.
- GSDIonescu was incorrectly handling the end sample of the GS. This could lead to GS that end outside the data.
  (https://github.com/mobilise-d/mobgap/pull/145)

### Removed

- When data is loaded, the error/warning that none of the sensors where available is removed 
  (https://github.com/mobilise-d/mobgap/pull/125)


## [0.3.0] - 2024-04-23

### Added

- All dataset class have a new property called `data_ss` referring to the single sensor that should be used in the
  algorithms.
  This was added to move all data related config (i.e. which sensor to use in a pipeline) to the dataset class, making
  it easier to implement dataset agnostic pipelines (https://github.com/mobilise-d/mobgap/pull/119)
- A evaluation pipeline for GSD (https://github.com/mobilise-d/mobgap/pull/124)
- ML based LR classification (https://github.com/mobilise-d/mobgap/pull/106)
- A evaluation/optimization pipeline for LRC (https://github.com/mobilise-d/mobgap/pull/106)

### Changed

- The loaded reference data now has stricter dtypes (https://github.com/mobilise-d/mobgap/pull/119)
- Renamed LRD (left-right-detection) to LRC (left-right-classification) (https://github.com/mobilise-d/mobgap/pull/106)
- For GSD and IC evaluation metrics it is now possible to configure what happens in case of 0-division 
  (https://github.com/mobilise-d/mobgap/pull/127)

### Fixed

- Loading the reference data from a trial without identified WBs will not raise an error anymore, but will correctly
  return an empty DataFrame (https://github.com/mobilise-d/mobgap/pull/119)
- We use operating system independent pandas dtypes everywhere (https://github.com/mobilise-d/mobgap/pull/118)
- Fixed issue with multi-groupby, that grouping by just a subset of the index cols resulted in cryptic error messages
  (https://github.com/mobilise-d/mobgap/pull/132)


### Development

- RTD previews are now build for PRs.

## [0.2.0] - 2024-03-22

### Added 

- Example Data fetching via Pooch (https://github.com/mobilise-d/mobgap/pull/113)

## [0.1.0] - 2024-03-22 

Initial Release

