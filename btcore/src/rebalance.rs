//! Rebalancing logic and signal processing

use chrono::{Datelike, NaiveDate};

/// Rebalance frequency options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RebalanceFreq {
    /// Rebalance every N trading days
    Days(usize),
    /// Rebalance on first trading day of each month
    Monthly,
    /// Rebalance on first trading day of each quarter
    Quarterly,
    /// Rebalance on first trading day of each year
    Yearly,
    /// Rebalance whenever signals change
    OnSignalChange,
}

impl Default for RebalanceFreq {
    fn default() -> Self {
        RebalanceFreq::Days(1)
    }
}

/// Find rebalance indices based on frequency
///
/// # Arguments
/// * `dates` - Trading dates
/// * `freq` - Rebalancing frequency
///
/// # Returns
/// Vector of indices where rebalancing should occur
pub fn find_rebalance_indices(dates: &[NaiveDate], freq: RebalanceFreq) -> Vec<usize> {
    if dates.is_empty() {
        return vec![];
    }

    match freq {
        RebalanceFreq::Days(n) => {
            (0..dates.len()).step_by(n).collect()
        }
        RebalanceFreq::Monthly => {
            find_period_start_indices(dates, |d| (d.year(), d.month()))
        }
        RebalanceFreq::Quarterly => {
            find_period_start_indices(dates, |d| (d.year(), (d.month() - 1) / 3))
        }
        RebalanceFreq::Yearly => {
            find_period_start_indices(dates, |d| d.year())
        }
        RebalanceFreq::OnSignalChange => {
            // Signal change detection is handled separately
            vec![0]
        }
    }
}

/// Find indices where a new period starts
fn find_period_start_indices<T: PartialEq>(
    dates: &[NaiveDate],
    period_fn: impl Fn(&NaiveDate) -> T,
) -> Vec<usize> {
    if dates.is_empty() {
        return vec![];
    }

    let mut indices = vec![0]; // Always include first date
    let mut current_period = period_fn(&dates[0]);

    for (i, date) in dates.iter().enumerate().skip(1) {
        let new_period = period_fn(date);
        if new_period != current_period {
            indices.push(i);
            current_period = new_period;
        }
    }

    indices
}

/// Find indices where signals change
///
/// # Arguments
/// * `signals` - 2D array of signals [time x assets]
///
/// # Returns
/// Vector of indices where any signal changes
pub fn find_signal_change_indices(signals: &[Vec<bool>]) -> Vec<usize> {
    if signals.is_empty() {
        return vec![];
    }

    let mut indices = vec![0]; // Always rebalance on first day

    for i in 1..signals.len() {
        if signals[i] != signals[i - 1] {
            indices.push(i);
        }
    }

    indices
}

/// Combine rebalance indices from frequency and signal changes
///
/// # Arguments
/// * `freq_indices` - Indices from frequency-based rebalancing
/// * `signal_indices` - Indices from signal changes
/// * `signal_priority` - If true, only rebalance on signal changes; else merge both
///
/// # Returns
/// Sorted, deduplicated vector of rebalance indices
pub fn merge_rebalance_indices(
    freq_indices: &[usize],
    signal_indices: &[usize],
    signal_priority: bool,
) -> Vec<usize> {
    if signal_priority {
        signal_indices.to_vec()
    } else {
        let mut merged: Vec<usize> = freq_indices
            .iter()
            .chain(signal_indices.iter())
            .copied()
            .collect();
        merged.sort_unstable();
        merged.dedup();
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dates(year: i32, month: u32, days: &[u32]) -> Vec<NaiveDate> {
        days.iter()
            .map(|&d| NaiveDate::from_ymd_opt(year, month, d).unwrap())
            .collect()
    }

    #[test]
    fn test_rebalance_days() {
        let dates = make_dates(2024, 1, &[2, 3, 4, 5, 8, 9, 10, 11, 12]);
        let indices = find_rebalance_indices(&dates, RebalanceFreq::Days(3));
        assert_eq!(indices, vec![0, 3, 6]);
    }

    #[test]
    fn test_rebalance_monthly() {
        let dates = vec![
            NaiveDate::from_ymd_opt(2024, 1, 15).unwrap(),
            NaiveDate::from_ymd_opt(2024, 1, 16).unwrap(),
            NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            NaiveDate::from_ymd_opt(2024, 2, 2).unwrap(),
            NaiveDate::from_ymd_opt(2024, 3, 1).unwrap(),
        ];
        let indices = find_rebalance_indices(&dates, RebalanceFreq::Monthly);
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_signal_change_indices() {
        let signals = vec![
            vec![true, false, true],
            vec![true, false, true],  // no change
            vec![true, true, true],   // change
            vec![true, true, true],   // no change
            vec![false, true, true],  // change
        ];
        let indices = find_signal_change_indices(&signals);
        assert_eq!(indices, vec![0, 2, 4]);
    }

    #[test]
    fn test_merge_indices() {
        let freq = vec![0, 5, 10];
        let signal = vec![0, 3, 7];

        let merged = merge_rebalance_indices(&freq, &signal, false);
        assert_eq!(merged, vec![0, 3, 5, 7, 10]);

        let signal_only = merge_rebalance_indices(&freq, &signal, true);
        assert_eq!(signal_only, vec![0, 3, 7]);
    }
}
