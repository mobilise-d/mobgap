# Digital Mobility Assessments (DMA) and Weekly Aggregation

The Mobilise-D Clinical Validation Study (CVS) established standardized procedures for aggregating gait parameters from multi-day free-living recordings into valid Digital Mobility Assessments (DMAs).

## Pipeline Capabilities

This pipeline provides **daily-level aggregations** (one row per recording). 
Users supply day-segmented raw IMU data, and the pipeline outputs aggregated DMOs including wear-time detection for each recording day. 
Unlike the original CVS workflow which relied on external, proprietary wear-time reports, 
MobGap incorporates transparent, high-performing wear-time detection algorithms directly into the pipeline.

## Mobilise-D DMA Requirements

According to Mobilise-D standards [1], a valid DMA requires:

- **Minimum 3 valid days** per participant
- **Valid day definition**: ≥12 hours of wear-time during waking hours (07:00–22:00)

Participants with fewer than 3 valid days are excluded from weekly-level DMAs.

The pipeline's daily `aggregated_parameters_` output includes `weartime_hours_during_waking` per recording, enabling users to extract end-to-end Mobilise-D DMOs without external dependencies or proprietary algorithms.

## Weekly Aggregation Example

We provide a complete example demonstrating the suggested DMO aggregation workflow per DMA: filtering days by weartime threshold, aggregating to weekly (per-participant) level, and applying the 3-day minimum requirement. 

See: `examples/aggregation/_98_dma_agg_pipeline_weartime_no_exc.py`

For demonstration purposes, you can generate synthetic multi-day recordings to illustrate the aggregation workflow without requiring actual CVS data.

## References

[1] Buekers, J., et al. Digital assessment of real-world walking in people with impaired mobility: How many hours and days are needed? *Int J Behav Nutr Phys Act*, 2025. 22(1)