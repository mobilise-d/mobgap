# Walking outcomes for clinical research

The main Mobilise-D pipeline provides two levels of walking outcomes:

- `per_wb_parameters_` contains one row per accepted walking bout, with its duration, step count, and average gait parameters.
- `aggregated_parameters_` summarises those bouts. By default, it contains one row for the recording passed to the pipeline.

This guide covers these outcomes from the default {class}`~mobgap.pipeline.MobilisedPipelineHealthy` and
{class}`~mobgap.pipeline.MobilisedPipelineImpaired` pipelines. It does not cover secondary algorithm outputs.
A walking bout is an episode assembled from accepted strides, rather than a geographical walking route.

The healthy pipeline is recommended for healthy adults, chronic obstructive pulmonary disease, and congestive heart
failure. The impaired pipeline is recommended for Parkinson's disease, multiple sclerosis, and proximal femoral fracture.
These are the cohort groupings assessed in the Mobilise-D technical validation studies.

## Parameters for each walking bout

The following quantities appear in `per_wb_parameters_`. A stride runs from one foot contact to the next contact of the
same foot. Both left and right strides can contribute to a bout.

| Quantity | Output column | Meaning and derivation |
| --- | --- | --- |
| Walking-bout duration | `duration_s` | Elapsed time in seconds between the first accepted stride's start and the last accepted stride's end. Calculated as `end - start`, divided by the sampling rate. This span can include short breaks allowed by bout assembly. |
| Walking-bout step count | `n_raw_initial_contacts` | Number of raw detected initial contacts within the final bout boundaries, including contacts on either boundary. This is the step-count input to aggregation. It includes contacts regardless of whether they formed an accepted stride. |
| Number of accepted strides | `n_strides` | Number of stride rows retained in the bout across both feet. This describes bout assembly and is not the step count. Do not obtain the step count by doubling this value. |
| Mean stride duration | `stride_duration_s` | Arithmetic mean of accepted stride durations, in seconds. Each stride duration is the difference between its end and start contact times. |
| Mean cadence | `cadence_spm` | Mean cadence across accepted strides, in steps per minute. The cadence algorithm detects its own step timings, smooths them, reports cadence per second, and maps those values onto strides before averaging within the bout. |
| Mean stride length | `stride_length_m` | Mean stride length across accepted strides, in metres. A biomechanical inverted-pendulum model uses vertical lower-back acceleration and sensor height to estimate length. Smoothed per-second estimates are mapped onto strides before averaging within the bout. |
| Mean walking speed | `walking_speed_mps` | Mean speed across accepted strides, in metres per second. Per-second speed is calculated as stride length × cadence / 120, then mapped onto strides and averaged within the bout. The mean speed need not equal the product of the two bout means divided by 120. |

The table also contains `start` and `end` as sample positions relative to the recording, and uses `wb_id` as its index.
`rule_name` and `rule_obj` identify the rule that ended the bout. These fields support identification and quality review;
they are not additional clinical outcomes.

The technical evidence for gait detection, contact detection, cadence, and stride length comes from
[Micó-Amigo et al., 2023][mico]. [Kirk et al., 2024][kirk] evaluates the combined pipeline, particularly walking speed.
The [mobgap paper][mobgap] reports the Python implementation and its revalidation. These studies establish measurement
performance in their assessed conditions. Clinical interpretation also requires the cohort-specific evidence below.

## Aggregations across walking bouts

The default pipelines return **25 aggregated columns**. The following inventory includes all of them, with exact
duration restrictions. It describes the implemented {class}`~mobgap.aggregation.MobilisedAggregator`.

Each input row is a walking bout. Unless stated otherwise, a mean gives each eligible bout equal weight; it does not
weight by bout duration or number of strides. Missing or implausible values are excluded as described below.

### All accepted walking bouts

| Outcome | Output column | Meaning and derivation |
| --- | --- | --- |
| Number of walking bouts | `wb_all__count` | Count of bouts with an eligible duration. |
| Total walking duration | `total_walking_duration_min` | Sum of accepted bout durations, converted from seconds to minutes. This is a conservative measure of retained walking, not a reliable estimate of all time spent walking. |
| Total walking-bout step count | `wb_all__n_raw_initial_contacts__sum` | Sum of the raw initial-contact counts within accepted bouts. |
| Typical bout duration | `wb_all__duration_s__avg` | **Median** bout duration, in seconds. Despite the `avg` suffix, this is not an arithmetic mean. |
| Upper-end bout duration | `wb_all__duration_s__p90` | 90th percentile of bout durations, in seconds. This describes longer bouts without selecting the single longest bout. |
| Variation in bout duration | `wb_all__duration_s__var` | Coefficient of variation of bout durations. Describes how much the duration differs between bouts. |
| Mean cadence | `wb_all__cadence_spm__avg` | Arithmetic mean of bout mean cadences, in steps per minute. |
| Mean stride duration | `wb_all__stride_duration_s__avg` | Arithmetic mean of bout mean stride durations, in seconds. |
| Variation in cadence | `wb_all__cadence_spm__var` | Coefficient of variation of bout mean cadences. Describes differences between bouts. |
| Variation in stride duration | `wb_all__stride_duration_s__var` | Coefficient of variation of bout mean stride durations. Describes differences between bouts. |

### Bouts longer than 10 seconds and up to 30 seconds

These use the restriction `10 < duration_s <= 30`.

| Outcome | Output column | Meaning and derivation |
| --- | --- | --- |
| Number of bouts in this duration range | `wb_10_30__count` | Count of eligible bouts in this range. This is an additional software output, outside the 24-DMO clinical-study panel. |
| Mean walking speed in this duration range | `wb_10_30__walking_speed_mps__avg` | Arithmetic mean of eligible bout mean speeds, in metres per second. |
| Mean stride length in this duration range | `wb_10_30__stride_length_m__avg` | Arithmetic mean of eligible bout mean stride lengths, in metres. |

### Bouts longer than 10 seconds

These use the restriction `duration_s > 10`, including bouts longer than 30 or 60 seconds.

| Outcome | Output column | Meaning and derivation |
| --- | --- | --- |
| Number of bouts longer than 10 seconds | `wb_10__count` | Count of eligible bouts longer than 10 seconds. |
| Upper-end walking speed | `wb_10__walking_speed_mps__p90` | 90th percentile of eligible bout mean speeds, in metres per second. |

### Bouts longer than 30 seconds

These use the restriction `duration_s > 30`.

| Outcome | Output column | Meaning and derivation |
| --- | --- | --- |
| Number of bouts longer than 30 seconds | `wb_30__count` | Count of eligible bouts longer than 30 seconds. |
| Mean walking speed | `wb_30__walking_speed_mps__avg` | Arithmetic mean of eligible bout mean speeds, in metres per second. |
| Mean stride length | `wb_30__stride_length_m__avg` | Arithmetic mean of eligible bout mean stride lengths, in metres. |
| Mean cadence | `wb_30__cadence_spm__avg` | Arithmetic mean of eligible bout mean cadences, in steps per minute. |
| Mean stride duration | `wb_30__stride_duration_s__avg` | Arithmetic mean of eligible bout mean stride durations, in seconds. |
| Upper-end walking speed | `wb_30__walking_speed_mps__p90` | 90th percentile of eligible bout mean speeds, in metres per second. |
| Upper-end cadence | `wb_30__cadence_spm__p90` | 90th percentile of eligible bout mean cadences, in steps per minute. |
| Variation in walking speed | `wb_30__walking_speed_mps__var` | Coefficient of variation of eligible bout mean speeds. |
| Variation in stride length | `wb_30__stride_length_m__var` | Coefficient of variation of eligible bout mean stride lengths. |

### Bouts longer than 60 seconds

| Outcome | Output column | Meaning and derivation |
| --- | --- | --- |
| Number of bouts longer than 60 seconds | `wb_60__count` | Count of eligible bouts with `duration_s > 60`. |

### Reading the statistics

- `p90` is a 90th percentile, not an observed maximum or a test of maximum walking capacity. Clinical publications may
  call it a "maximum" measure; check their definition.
- `var` is the coefficient of variation, calculated as sample standard deviation divided by the mean. Mobgap returns a
  dimensionless ratio, not statistical variance and not a percentage. Multiply by 100 to express it as a percentage.
  A coefficient of variation needs at least two usable bout values and a nonzero mean to be interpretable.
- `avg` is an arithmetic mean except for bout duration, where it is a median.
- Stride length remains in metres. Selecting the original Mobilise-D column names with `use_original_names=True` does
  not convert it to centimetres or convert coefficients of variation to percentages.
- Duration groups overlap. A bout longer than 60 seconds also contributes to the groups longer than 30 and 10 seconds.
  Their counts must not be added together to obtain the overall bout count.

[Koch et al., 2026][aggregation] explains the choice of restrictions and summary statistics and evaluates stability and
discrimination for the 24 selected daily and weekly DMOs. The extra `wb_10_30__count` column is excluded in mobgap's
{ref}`CVS aggregation example <sphx_glr_auto_examples_aggregation__99_cvs_agg_pipeline_no_exc.py>`, which states that it was not
officially verified as a DMO. Its availability in the software does not establish clinical validity.

Turn count is not part of this inventory. The main pipeline currently does not supply a per-bout `n_turns` column,
although the aggregator supports it. This missing connection is tracked in
[issue #275](https://github.com/mobilise-d/mobgap/issues/275).

## How the pipeline arrives at these outcomes

The pipeline processes the lower-back sensor recording in five main stages:

1. Detect candidate gait sequences, identifying sections of the signal that appear to contain walking.
2. Detect foot initial contacts within each sequence and assign them to the left or right foot. Estimate cadence,
   stride length, and walking speed per second.
3. Construct strides from the contact times and foot labels, then map the per-second gait estimates onto those strides.
4. Select strides with acceptable parameters and assemble them into final walking bouts using the walking-bout
   definition. Calculate each bout's duration, contact count, and mean gait parameters.
5. Check the bout parameters for plausibility and aggregate the eligible values across bouts.

```{figure} ../_static/images/mobilised_pipeline_kirk_2024_fig7.png
:alt: Published Mobilise-D pipeline diagram showing gait-sequence detection, contact detection, per-second gait estimation, stride interpolation, stride selection, and final walking-bout assembly for pipelines P1 and P2.
:width: 100%
:target: https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41598-024-51766-5/MediaObjects/41598_2024_51766_Fig7_HTML.png

The published Mobilise-D pipeline. Figure 7 from [Kirk et al., 2024](https://doi.org/10.1038/s41598-024-51766-5),
reproduced unchanged from the corrected article. © The Author(s) 2024, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Select the image to view it at full size.
```

The figure describes the original published pipeline. Current mobgap turn outputs differ from its final turn-output
description; see {ref}`clinical-output-turn-inclusion`.

The default cadence calculation uses its own contact detector, separate from the contact detector used to construct
strides and count steps. Bout mean cadence therefore need not equal the bout step count divided by bout duration.

### Why initial gait sequences and final walking bouts differ

An initial gait sequence is a candidate region identified from the sensor signal. Its boundaries depend on the gait
sequence detector. At this stage, the pipeline has not yet established which contacts and strides it can use, or
whether their parameters meet the walking-bout definition.

The main reason for constructing final walking bouts is to enforce the consensus-based, clinically motivated criteria
used in the [Mobilise-D walking-speed validation study][kirk]. These include limits on stride duration and length,
a minimum number of accepted strides, and the maximum allowed break between them. The initial gait-sequence detector
cannot enforce these criteria because the required contact times, foot labels, and stride parameters are not yet
available. Stride selection and bout assembly apply them once those quantities have been estimated. This gives the
final bouts a common definition even when different algorithms detect the initial sequences.

The pipeline takes a conservative approach to selecting periods from which it can estimate walking parameters.
Detecting movement as a gait sequence is only the first step. The pipeline must also detect usable initial contacts,
construct strides from their left-right labels, and obtain stride parameters that satisfy the selection rules.

If those later steps fail, part of an initial gait sequence can disappear from the final walking-bout results. A long
section of apparently continuous movement can therefore become several shorter bouts, or leave no accepted bout.
For the reported bout outcomes, the pipeline treats intervals without accepted gait as contributing no gait. This does
not establish that the person was not walking during those intervals.

Consequently, `total_walking_duration_min` measures the time covered by accepted bouts. It should not be treated as a
reliable measure of the person's overall amount of walking. The same coverage restriction affects the bout step count.
For an exploratory analysis requiring broader walking coverage, inspect the raw gait-sequence output, `gs_list_`,
before stride selection and bout assembly. Its summed duration has not been tuned or validated as an overall walking
amount outcome in this workflow either. Such an analysis needs its own validation for that purpose.

`per_wb_parameter_mask_` records plausibility checks on the bout parameters, using cohort, measurement condition, and
participant height. The per-bout table itself retains the values. The default aggregator excludes flagged values from
the corresponding summaries; invalid cadence or stride length also excludes that bout's speed from speed summaries.
The mask uses `True` for values within thresholds, `False` for values outside thresholds, and `NaN` when no check was
applied. A plausible value is not proof of accuracy or clinical validity. See
{func}`~mobgap.aggregation.apply_thresholds` for the checks.

### A recording summary is not automatically a daily or weekly outcome

The main pipeline does not check whether the supplied recording represents a full day, a short walking test, or an
excerpt from a longer recording. It also does not verify that the data covers the entire observation period intended
by the researcher. Users must check that the recording covers their intended start and end times and identify missing
periods or insufficient wear time before interpreting the aggregates.

The default aggregator pools the bouts in whatever recording it receives. Partial coverage can materially change
totals, bout-duration distributions, means, and percentiles, since the omitted periods may contain different walking
behaviour. A successfully completed pipeline run does not establish that the recording is complete or representative
of the intended observation period.

The default gait pipeline does not run wear-time detection, split a recording into valid days, or check whether
enough days were recorded. Run mobgap's wear-time detector alongside it to establish daily wear time, then apply the
daily and weekly criteria below.

The final CVS outcomes require a separate daily and weekly aggregation workflow, described below. Passing a seven-day
recording directly to the default main pipeline does not perform that workflow.

## Final seven-day aggregation for the clinical validation study

The [CVS protocol][cvs-protocol] includes a seven-day real-world mobility assessment at each study visit. Participants
were asked to wear the lower-back device continuously, 24 hours per day. The final workflow combines the daily and
weekly summaries described in the [aggregation framework][aggregation] with the valid-day and valid-week requirements
supported by [Buekers et al., 2025][weartime]. The
{ref}`CVS aggregation example <sphx_glr_auto_examples_aggregation__99_cvs_agg_pipeline_no_exc.py>` shows the processing
sequence for the study data.

### From walking bouts to one weekly result

1. Keep each participant and study visit separate. Assign the accepted bouts to their local calendar date and verify
   that the dates belong to the intended seven-day assessment. Check duplicate records, missing periods, and
   recordings extending beyond that window before aggregation.
2. Calculate the daily outcomes from the bouts in each full calendar day, using the duration restrictions and
   statistics listed above and the bout-parameter validity mask. For the clinical-study panel, retain the 24 selected
   DMOs and exclude the additional `wb_10_30__count` output.
3. Match each day's outcomes to its wear-time estimate, obtained from the detector or the original CVS wear-time
   report as described below. Retain a day only when wear time during 07:00 to 22:00 local time
   is **greater than 12 hours**. Exactly 12 hours does not meet this criterion. Exclude days with insufficient or
   unavailable wear-time information. This waking-hours window determines day eligibility; the DMOs themselves use
   walking bouts from the full calendar day.
4. Calculate the arithmetic mean of each daily DMO across the retained days, giving each day equal weight. Do not
   weight by the number of bouts or pool the week's bouts before calculating the summary.
5. Retain the participant-visit weekly result only when it contains **at least three valid days**. Record `n_days`
   alongside the outcomes. The general CVS recommendation does not require a minimum number of weekend days, and
   the valid days need not be consecutive within the seven-day assessment. Continue to aim for seven consecutive
   recording days to allow for non-wear and technical losses.

The seven-day assessment window and the minimum of three valid days are separate requirements. Three days collected
across unrelated weeks do not constitute the intended assessment. The main pipeline does not check either requirement.

### Obtaining daily wear time from the sensor recording

For analyses starting from raw sensor data, run {class}`~mobgap.weartime.WtdMegaritisSignal` through
{class}`~mobgap.weartime.pipeline.WtdEmulationPipeline` alongside the gait pipeline. Configure `waking_hours` as
07:00 to 22:00 and use `total_weartime_during_waking_min_`, divided by 60, as the daily waking wear time in hours.
Retain the daily DMO row only when this value is greater than 12.

Run the detector on individual local calendar days with correctly localized timestamps. Its waking-hours summary
rejects recordings that cross local midnight. Match the result to the participant, visit, and date used for the daily
DMO aggregation. This is wear time during any activity, including periods without walking. Do not use walking duration
or total wear time over the entire 24-hour day for the waking-hours eligibility check.

The {ref}`wear-time example <example_wtd_megaritis_signal>` demonstrates detection on a bundled lab recording and
includes a usage snippet for applying this valid-day rule to a user-supplied daily recording. It does not include a
bundled seven-day recording. The short lab trial alone cannot establish eligibility for a full-day assessment.

The original CVS aggregation example reads existing McRoberts wear-time reports. Use those original reports when
reproducing the original CVS exports exactly. For a new analysis using the mobgap detector, record the detector and
its configuration, since changing the wear-time method can change which days qualify.

### What the weekly numbers mean

Weekly aggregation takes the mean of the daily statistic for every DMO:

- For additive daily outcomes, such as step count, bout count, and walking duration, the weekly value is a **mean daily
  amount**. It is not the total amount observed over the week. For example, daily step counts of 3000, 5000, and 4000
  on three valid days produce a weekly result of 4000 steps per day. Missing days are not filled with zeros, and the
  result is not multiplied by seven.
- For daily means, such as walking speed in bouts longer than 30 seconds, the weekly value is the mean of the daily
  means. A day with many bouts has the same weight as a day with few bouts.
- For daily medians, 90th percentiles, and coefficients of variation, the weekly value is the mean of those daily
  statistics. It is not a median, percentile, or coefficient of variation recalculated from all bouts in the week.

Missing daily DMO values remain missing. The weekly mean uses the retained days with an available value for that DMO.
For example, a valid day can have no walking-speed estimate for bouts longer than 30 seconds if no eligible bouts of
that length occurred. The visit-level `n_days` count therefore does not establish that every DMO has that many usable
daily values. Report outcome-specific availability as well as the number of valid days.

When reproducing CVS exports, also match their reporting conventions. The CVS example uses the original column names,
converts stride length to centimetres and coefficients of variation to percentages, and applies its export rounding.
The main pipeline's default units remain metres and dimensionless ratios, as described above.

The wear-time thresholds are the general recommendation for studies covering multiple DMOs or conditions.
[Buekers et al., 2025][weartime] also provides outcome- and condition-specific requirements. A valid weekly aggregation
still describes conservatively accepted walking; it does not remove the coverage limitations of the underlying bouts.

## What these outputs are not suitable for

1. **Stride-by-stride comparisons of cadence, stride length, or walking speed.** These stride values are mapped from
   smoothed per-second estimates. Adjacent strides can share the same underlying estimates. They do not provide
   independent measurements for comparisons with individual reference strides or for detecting changes from one
   stride to the next. Stride duration is calculated from contact times rather than this interpolation, but that
   distinction does not establish validation of a custom stride-level analysis.
2. **Stride-to-stride variability or left-right asymmetry from the interpolated gait values.** The reported variability
   measures describe differences between bout means. They do not measure within-bout stride variability, and a left or
   right label does not turn the mapped length or speed into an independent foot-specific estimate.
3. **Concluding that rejected intervals contain no walking.** Initial-contact detection, left-right classification,
   and stride-parameter selection can remove parts of a detected gait sequence. The final bouts can be shorter or
   more fragmented than the underlying walking episode.
4. **A complete measure of walking time, all steps, or all physical activity.** Summed bout durations and step counts
   describe conservatively retained walking. Isolated steps, rejected intervals or bouts, other activities, and periods
   without usable recording are outside that coverage. These totals are not interchangeable with all walking time or
   a general-purpose pedometer total. Raw gait-sequence duration is an exploratory alternative, not a validated
   overall walking amount measure.
5. **Assuming the input covers the intended observation period.** The pipeline does not distinguish a full day from a
   short test or recording excerpt, or verify completeness. Users must check the intended duration and missing
   periods. Partial coverage can change both totals and the distribution of walking parameters. Use explicit valid-day
   and valid-week criteria before comparing participants or visits, following the [wear-time evidence][weartime].
6. **Measuring an instantaneous peak, maximum capacity, or route distance.** The speed percentiles summarise bout mean
   speeds. They do not give the fastest instant, a supervised capacity-test result, a geographical trajectory, or
   distance travelled along a route.
7. **Assuming equal accuracy in every bout or population.** Short bouts and slow walking are more difficult for the
   algorithms, particularly for stride length and speed. The technical studies describe these effects. Choose the
   pipeline for the studied cohort and assess changes to sensor placement, algorithms, or populations against the
   [technical validation evidence][mico] and [technical manual][technical-manual].
8. **Reading bout durations as uninterrupted walking episodes or turn counts as all turns.** Contact detection and
   left-right classification can be less reliable during turns, particularly sharp turns. Invalid or missing strides
   can split a continuous walking episode near a turn. This can shorten the mean or median bout duration and change
   the number of bouts. Restricting a turn analysis to accepted bouts can also preferentially exclude difficult turns.
9. **Treating every returned number as a clinically validated endpoint or diagnostic threshold.** Clinical evidence is
   specific to the outcome, population, aggregation, and research question. A software output, a plausibility flag,
   or a group association alone does not establish an individual diagnostic cutoff, a meaningful-change threshold,
   or responsiveness to treatment. The clinical studies below report different levels of support across outcomes.

(clinical-output-turn-inclusion)=
### Turn analyses need an explicit inclusion rule

An analysis that includes only turns within final accepted walking bouts can miss turns where contact detection,
left-right classification, or stride estimation failed. Sharp turns can therefore be underrepresented. Consider this
selection when interpreting turn counts, turn characteristics, and walking-bout durations.

In the current mobgap pipeline, `raw_turn_list_` is collected within the initially detected gait sequences, before
stride selection and final bout assembly. It is **not filtered to the final accepted walking bouts**. It can contain
turns outside those bouts, but still misses turns outside the detected gait sequences. The missing per-bout turn-count
connection in [issue #275](https://github.com/mobilise-d/mobgap/issues/275) must not be confused with this raw output.

For a dedicated analysis of turns independent of walking, researchers can apply
{class}`~mobgap.turning.TdElGohary` directly to the sensor recording, with the required sensor frame and channels.
Removing the gait-sequence restriction can recover relevant turning movements, but can also detect rotations during
non-walking activities, such as exercise. Define which activities count as turns for the research question and validate
that analysis. The detector's API documentation also describes current limitations in agreement with reference data.

## Validation evidence and implementation references

Technical validation asks how well a measurement agrees with a reference system. Aggregation research asks how to
summarise repeated measurements into stable, interpretable outcomes. Clinical validation asks whether the resulting
outcomes reflect relevant clinical constructs in a specified population. These sources address different parts of that
evidence; no single citation validates every column for every intended use.

### Technical measurement and software

- [Micó-Amigo, M. E., Bonci, T., Paraschiv-Ionescu, A., et al. Assessing real-world gait with digital technology?
  Validation, insights and recommendations from the Mobilise-D consortium. *Journal of NeuroEngineering and
  Rehabilitation* 20, 78, 2023.][mico] Compares gait detection, initial-contact detection, cadence, and stride-length
  algorithms across six cohorts. Supports cohort-specific algorithm selection and documents the effects of bout
  duration and walking speed on accuracy.
- [Kirk, C., Küderle, A., Micó-Amigo, M. E., et al. Mobilise-D insights to estimate real-world walking speed in multiple
  conditions with a wearable device. *Scientific Reports* 14, 1754, 2024.][kirk] Evaluates the combined Mobilise-D
  pipelines and walking-speed estimation. Use alongside the algorithm-level study.
- [Kirk, C., Küderle, A., Tasca, P., et al. Mobgap: A State-of-the-Art Python Framework for Reproducible Estimation and
  Algorithm Validation of Digital Mobility Outcomes from a Single Wearable Device. *Sensors* 26, 4294, 2026.][mobgap]
  Describes mobgap and revalidates its Python implementation against reference data. Report the software version and
  pipeline configuration with a study's results.

### Aggregation and data sufficiency

- [Koch, S., Buekers, J., Alvarez-Riu, P., et al. Aggregation framework for continuous digital health measures and
  application to walking activity data. *npj Digital Medicine*, 2026.][aggregation] Develops a framework combining
  technical and clinical considerations, statistical stability, discrimination, and expert judgement to select 24
  daily and weekly walking outcomes.
- [Buekers, J., Chernova, J., Koch, S., et al. Digital assessment of real-world walking in people with impaired
  mobility: How many hours and days are needed? *International Journal of Behavioral Nutrition and Physical Activity*
  22, 148, 2025.][weartime] Supports wear-time and valid-day requirements for weekly outcomes, with recommendations
  for individual outcomes and conditions.

### Clinical studies

- [Mikolaizak, A. S., Rochester, L., Maetzler, W., et al. Connecting real-world digital mobility assessment to clinical
  outcomes for regulatory and clinical endorsement–the Mobilise-D study protocol. *PLOS ONE* 17, e0269615,
  2022.][cvs-protocol] Describes the clinical validation study design and planned assessments. It is a protocol, not
  evidence that all planned outcomes passed validation.
- [Megaritis, D., Long, M., de Las Heras, M., et al. The construct validity of real-world digital mobility outcomes in
  people with COPD. *ERJ Open Research* 12, 00993-2025, 2026.][copd] Reports construct-validity support for 17 of the
  24 assessed DMOs in COPD.
- [Brittain, G., Long, M., Buckley, E., et al. Testing the construct validity of real-world digital mobility outcomes
  in multiple sclerosis. *Multiple Sclerosis Journal*, 2026.][ms] Reports construct-validity support for 19 of the
  24 assessed DMOs in multiple sclerosis.
- [Eckert, T., Berge, M. A., Long, M., et al. Construct validity of real-world digital mobility outcomes in patients
  after proximal femoral fracture: a cross-sectional observational study. *Scientific Reports* 16, 9535,
  2026.][pff] Reports construct-validity support for 17 of 24 DMOs in non-acute patients after fracture. Evidence in
  the acute phase is more limited, and support differs across domains.
- [Yarnall, A. J., Alcock, L., Schlenstedt, C., et al. The future of clinical trials? Monitoring mobility as an outcome
  measure in Parkinson's disease: The Mobilise-D study. *Journal of Parkinson's Disease* 16, 110–124,
  2026.][pd] Describes the 24-DMO panel in Parkinson's disease and controls, including differences across disease
  stages and recording feasibility. This is relevant clinical evidence, rather than a blanket validation of every
  endpoint property.

### Implementation manuals

- [Del Din, S., Kirk, C., Tasca, P., et al. Mobilise-D Technical Implementation Manual: How to validate and measure
  digital mobility outcomes from a wearable device. Version 1.0, 2025.][technical-manual] Guidance on sensor
  requirements, data collection, technical validation, and interpretation of measurement performance.
- [Rooks, D., Neatrour, I., Erdemli, G., et al. Mobilise-D Clinical Research Implementation Manual.
  2026.][clinical-manual] Practical guidance for clinical teams planning and running studies using digital mobility
  outcomes, including collection of meaningful, high-quality real-world mobility data.

[mico]: https://doi.org/10.1186/s12984-023-01198-5
[kirk]: https://doi.org/10.1038/s41598-024-51766-5
[mobgap]: https://doi.org/10.3390/s26134294
[aggregation]: https://doi.org/10.1038/s41746-026-03222-z
[weartime]: https://doi.org/10.1186/s12966-025-01851-3
[cvs-protocol]: https://doi.org/10.1371/journal.pone.0269615
[copd]: https://doi.org/10.1183/23120541.00993-2025
[ms]: https://doi.org/10.1177/13524585261464537
[pff]: https://doi.org/10.1038/s41598-026-43297-y
[pd]: https://doi.org/10.1177/1877718X251407026
[technical-manual]: https://doi.org/10.5281/zenodo.14901135
[clinical-manual]: https://zenodo.org/records/22834451
