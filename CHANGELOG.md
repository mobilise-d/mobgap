# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (+ the Migration Guide),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

