"""From windows (overlapping or not) to sample predictions using majority voting."""

import numpy as np
import pandas as pd
import warnings


def overlapping_windows_to_sample_labels(  # noqa: C901, PLR0912, PLR0915
    predictions: list[int],
    data_len: int,
    window_size: int = 500,
    stride: int = 125,
    extend_tail: bool = True,
    sampling_rate_hz: int = 100,
    min_confidence_short_bouts: float = 0.90,
    short_bout_threshold_minutes: float = 20,
    min_bout_duration_seconds: float = 15,
) -> tuple[pd.DataFrame, int, float, float, float, dict]:
    """
    Convert windowed predictions to per-sample labels using majority voting.

    Applies confidence threshold for short wear bouts not at data boundaries.
    Also removes short non-wear bouts by merging them into adjacent wear periods.

    Parameters
    ----------
    predictions : list or array
        Binary predictions (0 or 1) for each window
    data_len : int
        Total length of data in samples
    window_size : int, optional
        Window size in samples (default: 500 for 5s at 100Hz)
    stride : int, optional
        Stride between windows in samples (default: 125 for 75% overlap)
    extend_tail : bool, optional
        If True, extend the last window's prediction to cover any uncovered tail samples
    sampling_rate_hz : int, optional
        Sampling rate in Hz (default: 100)
    min_confidence_short_bouts : float, optional
        Minimum proportion of "wear" votes required for short bouts (default: 0.90 = 90%)
    short_bout_threshold_minutes : float, optional
        Duration threshold in minutes below which confidence filter applies (default: 20)
    min_bout_duration_seconds : int, optional
        Minimum bout duration in seconds for both wear and non-wear (default: 15)

    Returns
    -------
    pred_wt_df : pandas DataFrame
        Predicted wear time segments with 'start' and 'end' columns, indexed by 'wt_id'.
    total_weartime_samples : int
        Total wear time in samples
    total_weartime_seconds : float
        Total wear time in seconds
    total_weartime_minutes : float
        Total wear time in minutes
    total_weartime_hours : float
        Total wear time in hours
    coverage_info : dict
        Information about sample coverage

    Notes
    -----
    In this function we incorporate a post processing step assessing weartime bouts
    shorter than 20 minutes. These bouts are only kept as wear if the confidence
    (proportion of wear votes) is above the specified threshold (default 90%).
    This helps to reduce false positives from short wear time predictions that may be
    less reliable, while still allowing short wear bouts at the beginning or end of
    the data to be retained without filtering. Finally, we remove bouts which are
    shorter than 15 seconds because attaching and removing a device may be borderline
    possible within 15 seconds.
    """
    # Initialize vote counter for each sample
    vote_counts = np.zeros((data_len, 2), dtype=np.int32)  # [count_0, count_1]

    # Track the last covered sample
    last_covered_idx = -1

    # For each prediction window
    for window_idx, prediction in enumerate(predictions):
        # Calculate start position of this window
        start_idx = window_idx * stride
        end_idx = min(start_idx + window_size, data_len)

        # Add votes for all samples in this window
        if prediction == 0:
            vote_counts[start_idx:end_idx, 0] += 1
        else:
            vote_counts[start_idx:end_idx, 1] += 1

        last_covered_idx = max(last_covered_idx, end_idx - 1)

    # Apply majority voting: label is 1 if more votes for 1, else 0
    # We default to wear (1) on ties
    sample_labels = (vote_counts[:, 1] >= vote_counts[:, 0]).astype(np.int32)

    # Check for uncovered tail
    uncovered_samples = data_len - (last_covered_idx + 1)

    if extend_tail and uncovered_samples > 0 and len(predictions) > 0:
        # Extend the last window's prediction to the tail
        last_prediction = predictions[-1]
        sample_labels[last_covered_idx + 1 :] = last_prediction

    # Calculate coverage statistics
    total_votes = vote_counts[:, 0] + vote_counts[:, 1]
    coverage_info = {
        "total_samples": data_len,
        "covered_samples": int(np.sum(total_votes > 0)),
        "uncovered_samples": uncovered_samples,
        "last_covered_idx": last_covered_idx,
        "max_uncovered_tail": uncovered_samples if uncovered_samples > 0 else 0,
        "extended": extend_tail and uncovered_samples > 0,
    }

    # Remove short WEAR bouts and apply confidence filter
    wear_segments = []
    short_bout_threshold_samples = short_bout_threshold_minutes * 60 * sampling_rate_hz
    min_bout_samples = min_bout_duration_seconds * sampling_rate_hz

    # Check if there are any wear time samples at all
    if np.any(sample_labels == 1):
        # Find transitions
        # Pad with 0 at boundaries to detect edges
        padded = np.pad(sample_labels, (1, 1), constant_values=0)
        diff = np.diff(padded)

        # Start of wear segments (0->1 transition)
        starts = np.where(diff == 1)[0]
        # End of wear segments (1->0 transition)
        ends = np.where(diff == -1)[0]

        # Process each segment with confidence filtering
        for start, end in zip(starts, ends):
            bout_duration_samples = end - start
            bout_duration_minutes = bout_duration_samples / (sampling_rate_hz * 60)

            # Filter out very short bouts
            if bout_duration_samples < min_bout_samples:
                # Mark samples as non-wear since bout is too short
                sample_labels[start:end] = 0
                continue  # Skip to next bout

            # Check if this is a short bout
            is_short_bout = bout_duration_samples < short_bout_threshold_samples

            # Check if bout is at beginning or end of data
            at_boundary = (start == 0) or (end == data_len)

            # Apply confidence filter for short bouts not at boundaries
            if is_short_bout and not at_boundary:
                # Calculate confidence: proportion of wear votes in this bout
                bout_vote_counts = vote_counts[start:end]
                total_bout_votes = bout_vote_counts[:, 0] + bout_vote_counts[:, 1]

                # Avoid division by zero
                if np.sum(total_bout_votes) > 0:
                    wear_vote_proportion = np.sum(bout_vote_counts[:, 1]) / np.sum(total_bout_votes)
                else:
                    wear_vote_proportion = 0.0

                # Only keep this bout if it meets confidence threshold
                if wear_vote_proportion >= min_confidence_short_bouts:
                    wear_segments.append(
                        {
                            "start": start,
                            "end": end,
                            "duration_min": bout_duration_minutes,
                            "confidence": wear_vote_proportion,
                            "filtered": False,
                        }
                    )
                else:
                    # Mark samples as non-wear since bout didn't meet confidence threshold
                    sample_labels[start:end] = 0
            else:
                # Long bout or at boundary - keep it
                # Calculate confidence for reporting
                bout_vote_counts = vote_counts[start:end]
                total_bout_votes = bout_vote_counts[:, 0] + bout_vote_counts[:, 1]
                if np.sum(total_bout_votes) > 0:
                    wear_vote_proportion = np.sum(bout_vote_counts[:, 1]) / np.sum(total_bout_votes)
                else:
                    wear_vote_proportion = 1.0

                wear_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "duration_min": bout_duration_minutes,
                        "confidence": wear_vote_proportion,
                        "filtered": False,
                    }
                )

    # Remove short NON-WEAR bouts (merge into adjacent wear)
    # Only remove non-wear bouts that are sandwiched between two wear bouts
    # Boundary non-wear bouts (start or end of data) are kept
    if np.any(sample_labels == 0):
        nonwear_labels = 1 - sample_labels
        padded_nw = np.pad(nonwear_labels, (1, 1), constant_values=0)
        diff_nw = np.diff(padded_nw)
        nw_starts = np.where(diff_nw == 1)[0]
        nw_ends = np.where(diff_nw == -1)[0]

        for nw_start, nw_end in zip(nw_starts, nw_ends):
            bout_duration_samples = nw_end - nw_start

            # Only remove if short AND between wear periods
            at_boundary = (nw_start == 0) or (nw_end == data_len)
            if bout_duration_samples < min_bout_samples and not at_boundary:
                # Merge into wear by setting to 1
                sample_labels[nw_start:nw_end] = 1

        # Rebuild wear_segments after non-wear merging
        wear_segments = []
        if np.any(sample_labels == 1):
            padded = np.pad(sample_labels, (1, 1), constant_values=0)
            diff = np.diff(padded)
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            for start, end in zip(starts, ends):
                bout_duration_minutes = (end - start) / (sampling_rate_hz * 60)
                wear_segments.append(
                    {
                        "start": start,
                        "end": end,
                        "duration_min": bout_duration_minutes,
                    }
                )

    # Recalculate wear time statistics after all filtering
    wear_samples = np.sum(sample_labels == 1)
    total_weartime_samples = int(wear_samples)
    total_weartime_seconds = wear_samples / sampling_rate_hz
    total_weartime_minutes = total_weartime_seconds / 60.0
    total_weartime_hours = total_weartime_seconds / 3600.0

    # Calculate waking hours weartime (07:00-22:00) from post-processed sample_labels
    waking_start = int(7 * 3600 * sampling_rate_hz)
    waking_end = int(22 * 3600 * sampling_rate_hz)
    recording_hours = data_len / (3600 * sampling_rate_hz)

    # Check if recording is according to mobgap use case (single day)
    if data_len < waking_end:
        # Recording shorter than 22:00 - use full weartime
        warnings.warn(
            f"Recording duration ({recording_hours:.1f}h) is shorter than waking hours window (07:00-22:00). "
            f"Using total_weartime_hours for weartime_during_waking_hours.",
            stacklevel=2
        )
        total_weartime_hours_during_waking_ = total_weartime_hours
    elif recording_hours > 25:
        # Recording longer than 25 hours - use full weartime
        warnings.warn(
            f"Recording duration ({recording_hours:.1f}h) exceeds a full day. "
            f"Waking hours calculation assumes the recording is segmented per day. "
            f"Using total_weartime_hours for weartime_during_waking_hours.",
            stacklevel=2
        )
        total_weartime_hours_during_waking_ = total_weartime_hours
    else:
        # Normal day (22-25h): crop to waking hours
        sample_labels_waking = sample_labels.copy()
        sample_labels_waking[:waking_start] = 0
        sample_labels_waking[waking_end:] = 0

        total_weartime_hours_during_waking_ = np.sum(sample_labels_waking) / (3600 * sampling_rate_hz)

    # Convert to DataFrame
    if wear_segments:
        pred_wt_df = pd.DataFrame(wear_segments)[["start", "end"]]
    else:
        # Create empty DataFrame with correct columns
        pred_wt_df = pd.DataFrame(columns=["start", "end"])

    pred_wt_df.index.name = "wt_id"

    return (
        pred_wt_df,
        total_weartime_samples,
        total_weartime_seconds,
        total_weartime_minutes,
        total_weartime_hours,
        total_weartime_hours_during_waking_,
        coverage_info,
    )


def remove_isolated_short_periods(
    weartime_flags: np.ndarray, min_period_sec: float = 15.0, sampling_rate_hz: float = 100.0
) -> np.ndarray:
    """
    Remove isolated wear/non-wear periods shorter than minimum duration.

    Rationale: Device attachment/removal requires a minimum physical time. Periods shorter
    than this threshold (default 15 seconds) are likely sensor artifacts, voting edge effects,
    or brief environmental disturbances rather than true wear state changes.

    This post-processing step removes sequentially:
    1. Brief isolated WEAR periods (<15s) — removed first, as pooled model evaluation
       consistently showed FP > FN, indicating a systematic tendency to over-detect wear.
    2. Brief isolated NON-WEAR periods (<15s) — removed second, reflecting the same
       physical impossibility of device attachment/removal within this timeframe.

    Boundary periods (at the start or end of the recording) are exempt from removal
    in both stages, as these may represent genuine partial wear or non-wear periods
    truncated by the recording window.

    Parameters
    ----------
    weartime_flags : np.ndarray
        Binary flags (1=wear, 0=non-wear) from majority voting
    min_period_sec : float
        Minimum period duration in seconds (default: 15.0)
        Periods shorter than this will be removed
    sampling_rate_hz : float
        Sampling frequency in Hz (default: 100.0)

    Returns
    -------
    np.ndarray
        Flags with brief isolated periods removed
    """
    min_samples = int(min_period_sec * sampling_rate_hz)
    smoothed_flags = weartime_flags.copy()

    # Step 1: Remove short WEAR bouts
    if np.any(smoothed_flags == 1):
        padded = np.pad(smoothed_flags, (1, 1), constant_values=0)
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            bout_duration = end - start
            at_boundary = (start == 0) or (end == len(weartime_flags))
            if bout_duration < min_samples and not at_boundary:
                smoothed_flags[start:end] = 0

    # Step 2: Remove short NON-WEAR bouts
    if np.any(smoothed_flags == 0):
        nonwear = 1 - smoothed_flags
        padded_nw = np.pad(nonwear, (1, 1), constant_values=0)
        diff_nw = np.diff(padded_nw)
        nw_starts = np.where(diff_nw == 1)[0]
        nw_ends = np.where(diff_nw == -1)[0]

        for nw_start, nw_end in zip(nw_starts, nw_ends):
            bout_duration = nw_end - nw_start
            at_boundary = (nw_start == 0) or (nw_end == len(weartime_flags))
            if bout_duration < min_samples and not at_boundary:
                smoothed_flags[nw_start:nw_end] = 1

    return smoothed_flags
