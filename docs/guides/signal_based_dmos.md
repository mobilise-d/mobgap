# Signal-based Digital Mobility Outcomes (SDMOs)

The technical validity of several primary DMOs was previously established by Buekers et al. [1], addressing critical
questions regarding:

- The minimum daily recording hours required for reliable daily estimates.
- The number of recording days needed to achieve stable weekly representations.

Within the Sustain project, a new group of 53 signal-based parameters generated during the Mobilise-D project were
evaluated. These outcomes aim to capture complementary aspects of daily living mobility across six walking domains:
Variability (22 SDMOs), Magnitude (13 SDMOs), Postural Control (6 SDMOs), Symmetry (5 SDMOs), Turns (4 SDMOs), and
Rhythm (3 SDMOs).

Non-turn parameters are computed per walking bout (WB). Turns are returned as individual records. To obtain SDMOs as a single measure (e.g., per trial/day/week),
the per-WB parameters are aggregated using various statistical methods including median, 10th percentile, 90th percentile and standard deviation.
In addition, the parameters can be summarised across different WB duration categories since short and long WBs might capture distinct aspects of mobility and offer different clinical insights.
The WB duration categories used for the aggregation were all WBs, WBs > 10 s, WBs 10–30 s, WBs > 30 s, WBs 30–60 s, and WBs > 60 s.

The `MobilisedSDMO` pipeline combines independent calculator objects to calculate the full set of SDMOs irrespective
of the cohort. The calculator list can be changed through the `calculators` parameter, which makes individual outcomes
easy to add, remove, or configure without modifying the composition pipeline. `SDMOAggregator` aggregates the per-WB
results and can be configured through its `metrics` and `duration_filters` parameters.

The clinical reliability and validity of these SDMOs differ by cohort; the current per-cohort recommendations are
listed below.

## Input conventions and interpretation

Inputs use body-frame acceleration in m/s² and angular velocity in degrees/s. No implicit conversion to g or radians
is applied. Signal outcomes use the continuous accepted WB, including turns and samples between retained strides.
SD uses sample standard deviation, `ddof=1`. Range is maximum minus minimum. Acceleration RMS removes only the mean;
gyro RMS uses raw angular velocity. RMS divides by N. Total acceleration RMS is the Euclidean norm of the three
anatomical axis RMS values and is NaN if any axis is unavailable or unconfigured. Axis/total ratios are fractions;
a zero total gives NaN ratios. Valid configured per-axis RMS remains available with partial input.

Stride CV is `100 * sample_SD / mean` in percent, pooling retained strides from both feet. Cadence, stride length and
speed are processed per-second estimates mapped to strides. Their CV describes variability of those mapped estimates;
interpolation and processing can suppress or alter it. Stride duration comes from same-foot IC boundaries directly.
Missing, spurious or misclassified ICs can distort its CV even after stride selection. Ten true 1 s intervals have
0% CV; losing one IC can leave eight 1 s intervals and one 2 s interval, giving 30% CV. The current default selection
can retain all these intervals. This limitation does not imply automatic repair or tighter duration thresholds.

`jerk_acc_*` and `jerk_gyr_*` are dimensionless natural-log LDLJ scores over the FULL accepted WB. Acceleration is
differentiated once, angular velocity twice, using raw body axes without additional filtering. Both normalize by
`max(abs(signal - mean(signal)))`; this removes constant offsets from the peak, including a constant gravity
projection, but changing projections and noise bandwidth still affect the result. Constant signals and fewer than
three samples give NaN. Negative values are expected.

**Duration warning:** duration can dominate actual smoothness differences when comparing different-length bouts.
The same repeated 1 Hz sine for 5/10/20 s gives acceleration LDLJ -6.20017/-7.58646/-8.97276 and angular LDLJ
-13.09348/-15.86607/-18.63866. Report duration alongside LDLJ and account for it when interpreting comparisons.

`jerk_ratio_ml_is` and `jerk_ratio_pa_is` instead use `10*log10(RMS(horizontal jerk)/RMS(vertical jerk))` in dB,
with raw first derivatives. The common RMS duration factor cancels, so the LDLJ duration-power warning does not apply
to these ratios. Window composition still affects them. Missing required axes produce no corresponding ratio;
zero vertical jerk or other nonfinite ratios give NaN. These differ intentionally from MATLAB's differences of LDLJ.

`SampleEntropy` defaults to 50 Hz internally. Resampling precedes entropy and the >200-sample threshold, so the
minimum is more than 4 s, not 200 steps. At this rate adjacent samples are 20 ms apart; `dim=2`, `r=0.15`,
population-SD tolerance and inclusive Chebyshev matching are retained. Both lengths use the same N-dim extendible
starting positions, excluding self matches. SampEn is `-ln(A/B)`, the ratio of matching pairs at lengths dim+1 and dim.
No dim matches gives NaN; dim matches without extensions give positive infinity; constants give zero. Nonfinite axes
and fewer than two extendible templates give NaN. Changing the sampling rate changes the physical time scale and
therefore the statistic. The internal rate is deliberately different from the supplied native-rate MATLAB estimator.
The corrected common pool can change existing Python values, independently of that rate difference.

## Regularity and symmetry

`RegularitySymmetry` measures repetition in the acceleration waveform independently of ICs, retained strides,
cadence or other outcome estimates. The normalized unbiased autocovariance and regularity definitions follow
[Moe-Nilssen and Helbostad (2004)](https://doi.org/10.1016/S0021-9290(03)00233-1).
The IS asymmetry outputs use the regularity ratio expressed as `abs(1 - step/stride)`, the absolute difference divided
by the mean attributed to [Kobsar et al. (2014)](https://doi.org/10.1016/j.gaitpost.2013.09.008), and the linear
`(stride - step)/2` formula from [van Gelder et al. (2021)](https://doi.org/10.3390/sym13091560).
`symmetry_k_is` is a fraction, without multiplication by 100.

The following signal-processing and peak-selection details are implementation choices beyond those outcome
definitions. The cited studies do not establish these particular windows, thresholds or correction rules:

1. Each body-axis acceleration is mean-centered. Autocovariance at lag k is divided by N-k overlapping pairs and
   normalized by its zero-lag value. The full accepted WB is used, with no additional acceleration filtering here.
2. A moving average smooths the autocovariance to suppress small competing peaks. The nominal span is
   `max(1, floor(0.2 * sampling_rate_hz))` samples for IS/PA and `max(1, floor(0.1 * sampling_rate_hz))` for ML.
   The default uses odd spans, reducing an even span by one, with shrinking symmetric windows at the edges.
   The alternative uses a centered pandas rolling mean with the nominal span and truncated edge windows.
3. Candidate peaks must be separated by at least `floor(sampling_rate_hz / 4) + 1` samples. IS/PA keep nonnegative
   peaks and use the first two corrected candidates as step and stride regularity. ML keeps negative troughs for
   step regularity, reporting their positive magnitude, and positive peaks for stride regularity.
4. The IS step lag supplies a timing reference for ML, falling back to PA when IS has no usable step peak.
   Before correction, ML step candidates earlier than half that lag and stride candidates earlier than 1.5 times
   that lag are excluded. With no timing reference these lower bounds are omitted.
5. With at least two candidates in a peak search, a correction radius is `ceil(0.2 * median(candidate spacing))`.
   Candidates within that radius of either array boundary are excluded. Each remaining candidate is moved to
   the maximum of the unsmoothed curve from one radius before it through half a radius after it, rounding the
   latter down. For negative ML troughs, the sign-reversed curve is used.
6. Corrected candidates less than one radius from the first candidate of their group are merged, retaining the
   strongest. If fewer than two candidates were initially found, correction is skipped and their smoothed
   positions and heights are retained. Thus that fallback uses a different amplitude representation.

These rules are retained as the current method. Their numerical choices are not claimed as independently
validated by the cited papers. In short or irregular bouts, selected peaks can represent a different period from
one step or stride. Normalized unbiased autocovariance can exceed [-1, 1], and values are not clipped; consequently
strict 0–1 regularity or -1–1 linear-asymmetry bounds are not guaranteed. Ratio-based asymmetry can become unstable
near zero denominators. These limitations should be considered when interpreting individual bouts.

## Spectral outcomes

`FrequencyAmplitudeWidth` standardizes each configured anatomical axis with sample SD and retains two-second
periodic Hamming segments, 50% overlap, no further detrending, and an FFT length of twice the next power of two of
the window length. Short signals use the available window. Total-length and window-length zero padding are both
valid Welch choices; padding changes the sampled grid, not the underlying finite-window resolution.

The strongest local PSD maximum in the inclusive search band is selected, with exact ties going to the lower
frequency. Boundary peaks use neighboring PSD samples outside the band to establish a local maximum. No six-bin
separation rule is needed when choosing one dominant peak. Default bands remain IS/PA 0.4–3.1 Hz and ML 0.15–1.6 Hz,
which differ from the cited paper's single 0.5–3.0 Hz band. Frequencies stay on the FFT grid and never leave the band.

`amplitude_*` is peak spectral concentration of standardized acceleration, in 1/Hz, not physical acceleration
amplitude. `freq_*` and `width_*` are in Hz. Width uses linear interpolation at both half-height crossings and is
NaN if a crossing is absent or outside the band. A pure tone has nonzero finite-window width. Missing/unconfigured
axes are omitted; constant, nonfinite or fewer-than-two-sample axes and absent peaks give NaN outputs. Each axis
works independently. No slopes or alternate spectral estimator are introduced.

## Harmonic-ratio methods

Two independently selectable calculators use every supplied stride's actual `[start, end)` sample interval relative
to the supplied signal. Both preserve nonzero offsets, gaps, overlapping left/right rows and input row order during
calculation. Neither reconstructs strides from alternating starts nor moves endpoints. In the full pipeline,
these are the retained stride rows within the accepted WB, expressed in WB-local coordinates.

`HarmonicRatio` retains the existing `harmonic_ratio_acc_{is,pa,ml}` names, spectral peak/trough extraction,
normalization and coefficient averaging. Its IS/PA targets use k/T, and it searches neighborhoods of plus/minus 1/T
for maxima in even slots and minima in odd slots, with slot 2 fixed to 1. The retained optional ML path uses 2k/T,
fixes slot 1 to 1 and reverses the extrema groups. It averages normalized coefficients across strides, then returns
even/odd for all axes, omitting missing coefficients independently. This differs from the conventional method.
The stride-boundary fix can change existing values; the spectral estimator is otherwise retained.

`HarmonicRatioFourier` follows the per-stride Fourier-amplitude definition described by
[Bellanca et al. (2013)](https://doi.org/10.1016/j.jbiomech.2012.12.008). Its distinct outputs are
`harmonic_ratio_fourier_acc_{is,pa,ml}`. Each stride is mean-centered and analyzed by an N-point DFT, with T=N/fs.
The first 20 amplitudes at k/T are used directly. IS/PA use even/odd, ML uses odd/even, and the individual stride
ratios are averaged with equal weight. No additional filtering, tapering, zero padding, resampling, frequency
search or endpoint adjustment is applied. This implements the harmonic definition without claiming to reproduce
all preprocessing or recording protocols of the paper.

All 20 harmonics must be strictly below Nyquist, requiring more than 40 samples per stride. Short, constant or
nonfinite stride-axis signals give undefined ratios and are omitted from that axis's mean; if none contributes,
the result is NaN. Exact positive numerator/zero denominator yields infinity and remains in the mean. Small
numerical denominators can produce very large finite ratios. No clipping or additional QC threshold is introduced.
Missing/unconfigured axes are omitted and empty stride lists give an empty table. Boundary accuracy, available
bandwidth and the number of contributing strides affect interpretation; software checks do not establish clinical
reliability on short bouts.

The current default remains `HarmonicRatio`. To calculate both methods in `MobilisedSDMO`, append the new calculator:

```python
from mobgap.signal_based import HarmonicRatioFourier, MobilisedSDMO

calculators = MobilisedSDMO.PredefinedParameters.default["calculators"] + (
    ("harmonic_ratio_fourier", HarmonicRatioFourier(acc_columns=["acc_is", "acc_pa", "acc_ml"])),
)
sdmos = MobilisedSDMO(calculators=calculators)
# Supply data and its WB-local stride start/end positions to sdmos.calculate(...).
```

The distinct output names allow both methods to coexist in WB results and aggregation. For a one-second stride
`sin(4*pi*t) + 0.4*sin(6*pi*t)`, Fourier IS/PA HR is 2.5. The retained peak/trough extraction gives approximately
25.1953 on 100 samples, demonstrating why these results are not interchangeable.

## Aggregation and validation limits

`SDMOAggregator` summarizes the distribution of per-WB values with equal weight per contributing bout and NaNs
omitted separately for each metric. With finite observations, medians and percentiles need at least one contributor; sample SD needs
two. Groups without contributing values return NaN. Duration groups use exclusive lower and inclusive upper bounds.
These statistics are bout-summary distributions, not pooled whole-recording features. The pipeline does not pass
primary-DMO validity masks to SDMO aggregation. Setting `dmo_aggregation=None` allows configured SDMO aggregation
to proceed independently; disabling both aggregators gives `aggregated_parameters_=None`. A turn-only configuration
has no WB outcomes to aggregate. No new QC thresholds, missing-IC repair or clinical weighting are applied.

The historical cohort recommendations below describe research outcomes. Passing software checks does not establish
clinical validity for these modified estimators or new cohorts, placements or protocols. The regularity/symmetry
algorithm is retained, with its additional implementation choices documented above. Both harmonic-ratio definitions
and their stride-window behavior are documented above.
The supplied MATLAB PCI, step outcomes, slopes, missing helper and other MATLAB-only findings are outside this
implementation.

## Relationship to the supplied MATLAB reference

Python retains mean removal for acceleration RMS; the supplied MATLAB `detrend` removes a linear trend. Dimensional
acceleration outcomes use m/s² here, so values from reference inputs expressed in g require explicit conversion.
Raw gyro RMS, sample SD, acceleration range, current LDLJ formulas and individual turn formulas are retained.
Python stride-duration CV uses the same retained duration records for SD and mean, avoiding the supplied MATLAB
step-versus-stride SD mismatch. The MATLAB ML spectral fallback can return an out-of-band frequency; that behavior
is not reproduced. Both FFT padding choices are valid, while the retained periodic Hamming convention and
interpolated width extraction can produce different numerical spectral results.

The entropy common-pool correction is applied in Python only. Python deliberately retains population-SD tolerance
and default 50 Hz resampling, whereas the supplied MATLAB estimator uses sample SD and the native input rate.
The RMS jerk ratios deliberately remain different from MATLAB's differences of LDLJ scores. Per-turn values are
reported without inferring an aggregation rule from an unavailable MATLAB caller. No MATLAB files are repaired;
MATLAB-only outcomes remain outside this change.

## Turn outcomes

`TurnSDMO.calculate` consumes one already-sliced, correctly oriented turn window and returns a
`signal_based_parameters_` Series indexed by parameter name. The caller selects half-open start/end intervals,
keeps turn identity and metadata, and assembles the per-turn table. No per-WB turn percentage or automatic
recording-level turn summary is calculated.

The parameters are `turn_mean_ang_vel = abs(mean(yaw))`, `turn_peak_ang_vel = max(abs(yaw))`, both in degrees/s,
and `turn_smoothness`, angular LDLJ over the entire supplied turn. Mean yaw is not the mean of absolute yaw when
its sign changes. Smoothness is dimensionless, may be negative and depends strongly on turn duration. It needs
at least three samples and is undefined for constant yaw. Empty data returns an empty Series.

The existing `MobilisedSDMO` composition keeps its bulk `per_turn_parameters_` table, with metadata, by slicing
turn windows before calling individual calculators. This table is assembled by the caller, not by `TurnSDMO`.

The full pipeline exposes the same `per_turn_parameters_` with recording-local intervals and the existing
`(gs_id, turn_id)` detector index. It retains every turn already detected within gait sequences, including turns
crossing or outside accepted WBs. Nullable `wb_id` is present only for a unique fully containing accepted WB.
This does not extend turn detection to the whole recording. Disabling SDMO calculation gives None; disabling the
turn calculator or having no turns gives an empty table. Raw turns never enter `SDMOAggregator`.

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
- `wb_30_60__angular_acceleration_gyr_is__p90`
- `wb_30_60__range_acc_pa__std`
- `wb_60__range_acc_pa__std`
- `wb_30_60__range_acc_ml__std`
- `wb_30__range_acc_ml__p90`
- `wb_30__range_acc_ml__median`
- `wb_60__range_acc_ml__p10`
- `wb_30__sd_gyr_ml__std`
- `wb_30__stride_regularity_is__std`

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
