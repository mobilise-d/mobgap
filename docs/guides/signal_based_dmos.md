# Signal-based Digital Mobility Outcomes (SDMOs)

The technical validity of several primary DMOs was previously established by Buekers et al. [1], addressing critical
questions regarding:

- The minimum daily recording hours required for reliable daily estimates.
- The number of recording days needed to achieve stable weekly representations.

Within the Sustain project, a new group of 53 signal-based parameters generated during the Mobilise-D project were
evaluated. These outcomes aim to capture complementary aspects of daily living mobility across six walking domains:
Variability (22 SDMOs), Magnitude (13 SDMOs), Postural Control (6 SDMOs), Symmetry (5 SDMOs), Turns (4 SDMOs), and
Rhythm (3 SDMOs).

These 53 parameters are computed per walking bout (WB). To obtain SDMOs as a single measure (e.g., per trial/day/week),
the per-WB parameters are aggregated using various statistical methods including median, 10th percentile, 90th percentile and standard deviation.
In addition, the parameters can be summarised across different WB duration categories since short and long WBs might capture distinct aspects of mobility and offer different clinical insights.
The WB duration categories used for the aggregation were all WBs, WBs > 10 s, WBs 10–30 s, WBs > 30 s, WBs 30–60 s, and WBs > 60 s.

The `MobilisedSDMO` pipeline combines independent calculator objects to calculate the full set of SDMOs irrespective
of the cohort. The calculator list can be changed through the `calculators` parameter, which makes individual outcomes
easy to add, remove, or configure without modifying the composition pipeline. `SDMOAggregator` aggregates the per-WB
results and can be configured through its `metrics` and `duration_filters` parameters.

The clinical reliability and validity of these SDMOs differ by cohort; the current per-cohort recommendations are
listed below.

## Recommended SDMOs Per-Cohort

### MS
- `wb_10__amplitude_is__p90`
- `wb_30__amplitude_is__std`
- `wb_30__freq_pa__p10`
- `wb_30__freq_is__p10`
- `all__harmonic_ratio_acc_pa__median`
- `wb_10__harmonic_ratio_acc_pa__p10`
- `wb_30__harmonic_ratio_acc_is__std`
- `wb_30__jerk_acc_is__p10`
- `wb_10__rms_ratio_acc_is__p90`
- `wb_30__range_acc_pa__p10`
- `wb_30__step_regularity_is__p10`
- `wb_30__step_regularity_is__median`
- `wb_60__step_regularity_is__p90`
- `wb_10__stride_regularity_is__p90`
- `wb_30__stride_regularity_is__std`
- `wb_10__symmetry_k_is__p90`
- `wb_10__turn_dur_percentage_from_wb_dur__median`
- `wb_10__turn_dur_percentage_from_wb_dur__std`
- `wb_60__turn_dur_percentage_from_wb_dur__p90`

### PD
- `wb_30__amplitude_is__std`
- `wb_60__amplitude_is__std`
- `wb_10__jerk_acc_is__std`
- `wb_30__rms_total_acc__std`
- `wb_30__range_acc_pa__p90`
- `wb_30__range_acc_pa__median`
- `wb_60__range_acc_pa__std`
- `wb_30__range_acc_ml__p90`
- `wb_30__range_acc_ml__median`

### PFF
- `wb_30__amplitude_is__std`
- `wb_60__amplitude_is__std`
- `wb_30__freq_pa__p10`
- `wb_30__jerk_acc_is__p10`
- `wb_30__jerk_acc_is__median`
- `wb_60__jerk_acc_is__p10`
- `wb_30_60__jerk_gyr_is__p90`
- `wb_30_60__range_acc_pa__std`
- `wb_60__range_acc_pa__std`
- `wb_30_60__range_acc_ml__std`
- `wb_30__range_acc_ml__p90`
- `wb_30__range_acc_ml__median`
- `wb_60__range_acc_ml__p10`
- `wb_30__sd_gyr_ml__std`
- `wb_30__stride_regularity_is__std`
- `wb_60__turn_smoothness__median`
- `wb_10__turn_dur_percentage_from_wb_dur__median`
- `wb_30__turn_dur_percentage_from_wb_dur__median`
- `wb_30__turn_dur_percentage_from_wb_dur__std`

### COPD
- `wb_30__amplitude_is__std`
- `all__cv_stride_length_m__p10`
- `wb_30__rms_total_acc__p10`
- `wb_30__rms_total_acc__std`
- `wb_30__range_acc_pa__p10`
- `wb_30__range_acc_pa__p90`
- `wb_30__range_acc_pa__median`
- `wb_30__range_acc_ml__p90`
- `wb_30__stride_regularity_is__std`


## References

[1] Buekers, J., et al. Digital assessment of real-world walking in people with impaired mobility: How many hours and days are needed? *Int J Behav Nutr Phys Act*, 2025. 22(1)
