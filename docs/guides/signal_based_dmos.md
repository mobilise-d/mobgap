# Signal-based Digital Mobility Outcomes (SDMOs)

The technical validity of several primary DMOs was previously established by Buekers et al. [1], addressing critical
questions regarding:

- The minimum daily recording hours required for reliable daily estimates.
- The number of recording days needed to achieve stable weekly representations.

Within the Sustain project, a new group of 53 signal-based parameters generated during the Mobilise-D project were evaluated.
These outcomes aim to capture complementary aspects of daily living mobility across six walking domains: 
Variability (22 SDMOs), Magnitude (13 SDMOs), Postural Control (6 SDMOs), Symmetry (5 SDMOs), Turns (4 SDMOs), and Rhythm (3 SDMOs).

These 53 parameters are computed per walking bout (WB). To obtain SDMOs as a single measure (e.g., per trial/day/week),
the per-WB parameters are aggregated using various statistical methods including median, 10th percentile, 90th percentile and standard deviation.
In addition, the parameters can be summarised across different WB duration categories since short and long WBs might capture distinct aspects of mobility and offer different clinical insights.
The WB duration categories used for the aggregation were all WBs, WBs > 10 s, WBs 10–30 s, WBs > 30 s, WBs 30–60 s, and WBs > 60 s.

The `SDMO` block containing algorithms for signal-based parameter calculations and the custom aggregator (`SDMOAggregator`)
for these parameters are implemented to calculate the full set of SDMOs irrespective of the cohort of the data.
However, the clinical reliability and validity of these SDMOs for each cohort differ and our per-cohort recommendations are provided below.

It is important to note that the `SDMO` block and the `SDMOAggregator` is quite flexible to allow easy implementation of
new parameter calculations by adding a new method (e.g., `_calculate_my_params`) to `SDMO` or new aggregations by
providing `metrics` and/or `duration_filters` in initialising the `SDMOAggregator`.

## Recommended SDMOs Per-Cohort

### MS
- `wb_10__Amplitude_is__p90`
- `wb_30__Amplitude_is__std`
- `wb_30__Freq_pa__p10`
- `wb_30__Freq_is__p10`
- `wb_all__HarmonicRatio_acc_pa__median`
- `wb_10__HarmonicRatio_acc_pa__p10`
- `wb_30__HarmonicRatio_acc_is__std`
- `wb_30__Jerk_acc_is__p10`
- `wb_10__RMSRatio_acc_is__p90`
- `wb_30__Range_acc_pa__p10`
- `wb_30__StepRegularity_is__p10`
- `wb_30__StepRegularity_is__median`
- `wb_60__StepRegularity_is__p90`
- `wb_10__StrideRegularity_is__p90`
- `wb_30__StrideRegularity_is__std`
- `wb_10__Symmetry_K_is__p90`
- `wb_10__Turn_Dur_Percentage_From_WBDur__median`
- `wb_10__Turn_Dur_Percentage_From_WBDur__std`
- `wb_60__Turn_Dur_Percentage_From_WBDur__p90`

### PD
- `wb_30__Amplitude_is__std`
- `wb_60__Amplitude_is__std`
- `wb_10__Jerk_acc_is__std`
- `wb_30__RMSTotal_acc__std`
- `wb_30__Range_acc_pa__p90`
- `wb_30__Range_acc_pa__median`
- `wb_60__Range_acc_pa__std`
- `wb_30__Range_acc_ml__p90`
- `wb_30__Range_acc_ml__median`

### PFF
- `wb_30__Amplitude_is__std`
- `wb_60__Amplitude_is__std`
- `wb_30__Freq_pa__p10`
- `wb_30__Jerk_acc_is__p10`
- `wb_30__Jerk_acc_is__median`
- `wb_60__Jerk_acc_is__p10`
- `wb_30_60__Jerk_gyr_is__p90`
- `wb_30_60__Range_acc_pa__std`
- `wb_60__Range_acc_pa__std`
- `wb_30_60__Range_acc_ml__std`
- `wb_30__Range_acc_ml__p90`
- `wb_30__Range_acc_ml__median`
- `wb_60__Range_acc_ml__p10`
- `wb_30__SD_gyr_ml__std`
- `wb_30__StrideRegularity_is__std`
- `wb_60__Turn_Smoothness__median`
- `wb_10__Turn_Dur_Percentage_From_WBDur__median`
- `wb_30__Turn_Dur_Percentage_From_WBDur__median`
- `wb_30__Turn_Dur_Percentage_From_WBDur__std`

### COPD
- `wb_30__Amplitude_is__std`
- `wb_all__CV_stride_length_m__p10`
- `wb_30__RMSTotal_acc__p10`
- `wb_30__RMSTotal_acc__std`
- `wb_30__Range_acc_pa__p10`
- `wb_30__Range_acc_pa__p90`
- `wb_30__Range_acc_pa__median`
- `wb_30__Range_acc_ml__p90`
- `wb_30__StrideRegularity_is__std`


## References

[1] Buekers, J., et al. Digital assessment of real-world walking in people with impaired mobility: How many hours and days are needed? *Int J Behav Nutr Phys Act*, 2025. 22(1)