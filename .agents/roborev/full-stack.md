# Pipeline and library correctness review

{{ index .Includes "guidance" }}

Trace affected workflows from data loading and coordinate conversion through
individual algorithms, pipeline composition, gait-sequence iteration, stride
selection, walking-bout assembly, and aggregation. Check both independently usable
algorithms and supported end-to-end pipelines, including downstream consumers.

Preserve DataFrame column names, dtypes, index names and levels, ordering, region
identifiers, empty-result schemas, and missing-value behavior. Check sample-based
versus time-based values, recording-relative versus region-relative offsets,
nested-region aggregation, and each API's documented interval boundaries.

Inspect cloning and parameter ownership, input mutation, repeated execution,
caching, resource handling, optional dependencies, error propagation, and trust
boundaries where relevant. Check existing integration and regression coverage
and supplied results without executing anything. Leave detailed literature
comparison and scientific terminology to the scientific correctness reviewer.
