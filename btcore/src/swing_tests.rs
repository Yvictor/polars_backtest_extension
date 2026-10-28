use super::*;

fn free() -> SwingCosts {
    SwingCosts {
        fee_rate: 0.0,
        sell_tax: 0.0,
        short_sell_fee: 0.0,
        hedge_cost: 0.0,
        ..SwingCosts::default()
    }
}
fn bar(date: i32, price: f64) -> Bar {
    Bar {
        date,
        symbol: 0,
        open: price,
        high: price,
        low: price,
        close: price,
        limit_up: 200.0,
        limit_down: 1.0,
        dividend: 0.0,
    }
}
fn target(date: i32, weight: f64) -> Target {
    Target {
        date,
        symbol: 0,
        weight,
    }
}
fn sim(b: &[Bar], t: &[Target], c: SwingCosts, r: SwingRules) -> SwingResult {
    run_swing(b, t, 1_000_000.0, &c, &r, None, None).unwrap()
}
fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} != {b}");
}

#[test]
fn lot_sizing_is_at_signal_close_and_truncates_signed_shares() {
    for sign in [-1.0, 1.0] {
        let s = sim(
            &[bar(0, 123.0), bar(1, 130.0)],
            &[target(0, sign * 0.5)],
            free(),
            SwingRules::default(),
        );
        assert_eq!(s.fills[0].shares, sign as i64 * 4000);
        assert_eq!(s.fills[0].price, 130.0);
        assert_eq!(s.daily.len(), 2);
    }
}

#[test]
fn dividends_and_calendar_borrow_only_on_overnight_positions() {
    for sign in [-1.0, 1.0] {
        let b = [
            bar(0, 100.0),
            Bar {
                dividend: 9.0,
                ..bar(1, 100.0)
            },
            Bar {
                dividend: 2.0,
                ..bar(4, 98.0)
            },
        ];
        let s = sim(
            &b,
            &[target(0, sign * 0.5)],
            SwingCosts {
                borrow_rate: 0.365,
                ..free()
            },
            SwingRules::default(),
        );
        near(s.daily[1].costs, 0.0);
        near(s.daily[1].dividends, 0.0);
        near(s.daily[2].dividends, sign * 10_000.0);
        near(
            s.daily[2].borrow_cost,
            if sign < 0.0 { 1500.0 } else { 0.0 },
        );
        near(s.daily[2].pnl, if sign < 0.0 { -1500.0 } else { 0.0 });
    }
}

#[test]
fn calendar_borrow_widens_dates_before_subtraction() {
    let s = sim(
        &[
            bar(i32::MIN, 100.0),
            bar(i32::MIN + 1, 100.0),
            bar(i32::MAX, 100.0),
        ],
        &[target(i32::MIN, -0.5)],
        SwingCosts {
            borrow_rate: 0.365,
            ..free()
        },
        SwingRules::default(),
    );
    let days = i64::from(i32::MAX) - i64::from(i32::MIN + 1);
    assert_eq!(s.daily[1].borrow_cost, 0.0);
    assert_eq!(
        s.daily[2].borrow_cost,
        500_000.0 * 0.365 * days as f64 / 365.0
    );
    assert!(s.daily[2].borrow_cost > 0.0);
}

#[test]
fn every_sale_pays_tax_and_only_new_shorts_pay_surcharge() {
    let costs = SwingCosts::default();
    let s = sim(
        &[bar(0, 100.0), bar(1, 100.0), bar(2, 100.0), bar(3, 100.0)],
        &[target(0, 0.5), target(1, -0.5), target(2, 0.0)],
        costs,
        SwingRules::default(),
    );
    let a = &s.fills;
    near(a[0].cost, 500_000.0 * 0.001425 * 0.2);
    assert_eq!(a[1].shares, -9000); // equity after entry fee truncates the new short to 4 lots
    near(
        a[1].cost,
        900_000.0 * (0.001425 * 0.2 + 0.003) + 400_000.0 * 0.0008,
    );
    assert_eq!(a[2].shares, 4000);
    near(a[2].cost, 400_000.0 * 0.001425 * 0.2);
}

#[test]
fn auction_limits_block_nonlocked_bars_and_retry_next_session() {
    for (weight, up, down) in [(0.5, 100.0, 1.0), (-0.5, 200.0, 100.0)] {
        let locked_auction = Bar {
            high: 110.0,
            low: 90.0,
            limit_up: up,
            limit_down: down,
            ..bar(1, 100.0)
        };
        let s = sim(
            &[bar(0, 100.0), locked_auction, bar(2, 100.0)],
            &[target(0, weight)],
            free(),
            SwingRules::default(),
        );
        assert_eq!(s.fills.len(), 1);
        assert_eq!(s.fills[0].date, 2);
    }
}

#[test]
fn long_sales_and_short_covers_also_wait_at_the_limit() {
    for sign in [-1.0, 1.0] {
        let blocked = if sign > 0.0 {
            Bar {
                limit_down: 100.0,
                ..bar(2, 100.0)
            }
        } else {
            Bar {
                limit_up: 100.0,
                ..bar(2, 100.0)
            }
        };
        let s = sim(
            &[bar(0, 100.0), bar(1, 100.0), blocked, bar(3, 100.0)],
            &[target(0, sign * 0.5), target(1, 0.0)],
            free(),
            SwingRules::default(),
        );
        assert_eq!(s.fills.len(), 2);
        assert_eq!(s.fills[1].date, 3);
    }
}

#[test]
fn missing_bars_and_nan_auctions_preserve_pending_and_marks() {
    let b = [
        bar(0, 100.0),
        Bar {
            open: f64::NAN,
            ..bar(1, 100.0)
        },
        bar(2, 100.0),
        Bar {
            symbol: 1,
            ..bar(3, 200.0)
        },
        bar(4, 100.0),
    ];
    let s = sim(
        &b,
        &[target(0, 0.5), target(2, 0.0)],
        free(),
        SwingRules::default(),
    );
    assert_eq!(
        s.fills.iter().map(|f| f.date).collect::<Vec<_>>(),
        vec![2, 4]
    );
    near(s.daily[3].long, 500_000.0);
}

#[test]
fn borrow_checked_on_fill_day_and_reversal_closes_long_first() {
    let b: Vec<_> = (0..4).map(|d| bar(d, 100.0)).collect();
    let borrow = [Borrow {
        date: 3,
        symbol: 0,
        can_borrow: true,
    }];
    let s = run_swing(
        &b,
        &[target(0, 0.5), target(1, -0.5)],
        1e6,
        &free(),
        &SwingRules::default(),
        Some(&borrow),
        None,
    )
    .unwrap();
    assert_eq!(
        s.fills
            .iter()
            .map(|f| (f.date, f.shares))
            .collect::<Vec<_>>(),
        vec![(1, 5000), (2, -5000), (3, -5000)]
    );
}

#[test]
fn close_execution_borrow_denial_closes_long_and_retries_only_short() {
    let b: Vec<_> = (0..4)
        .map(|d| Bar {
            open: 90.0,
            ..bar(d, 100.0)
        })
        .collect();
    let borrow = [
        Borrow {
            date: 2,
            symbol: 0,
            can_borrow: false,
        },
        Borrow {
            date: 3,
            symbol: 0,
            can_borrow: true,
        },
    ];
    let s = run_swing(
        &b,
        &[target(0, 0.5), target(1, -0.5)],
        1e6,
        &free(),
        &SwingRules {
            execution_close: true,
            ..SwingRules::default()
        },
        Some(&borrow),
        None,
    )
    .unwrap();
    assert_eq!(
        s.fills
            .iter()
            .map(|f| (f.date, f.shares, f.price))
            .collect::<Vec<_>>(),
        vec![(1, 5000, 100.0), (2, -5000, 100.0), (3, -5000, 100.0)]
    );
    assert_eq!(s.daily[2].long, 0.0);
    assert_eq!(s.daily[2].short, 0.0);
    assert_eq!(s.daily[3].short, -500_000.0);
}

#[test]
fn new_target_cancels_an_unfilled_omitted_order() {
    let b = [
        bar(0, 100.0),
        Bar {
            limit_up: 100.0,
            ..bar(1, 100.0)
        },
        bar(2, 100.0),
    ];
    let s = sim(
        &b,
        &[
            target(0, 0.5),
            Target {
                date: 1,
                symbol: 1,
                weight: 0.0,
            },
        ],
        free(),
        SwingRules::default(),
    );
    assert!(s.fills.is_empty());
}

#[test]
fn strict_tp_crossing_gap_fill_and_adverse_slippage_both_sides() {
    for sign in [-1.0, 1.0] {
        let level = 100.0 * (1.0 + sign * 0.1);
        let touch = if sign > 0.0 {
            Bar {
                high: level,
                ..bar(2, 100.0)
            }
        } else {
            Bar {
                low: level,
                ..bar(2, 100.0)
            }
        };
        let cross = if sign > 0.0 {
            Bar {
                high: 112.0,
                ..bar(3, 100.0)
            }
        } else {
            Bar {
                low: 88.0,
                ..bar(3, 100.0)
            }
        };
        let r = SwingRules {
            take_profit: Some(0.1),
            ..SwingRules::default()
        };
        let s = sim(
            &[bar(0, 100.0), bar(1, 100.0), touch, cross],
            &[target(0, sign * 0.5)],
            free(),
            r,
        );
        assert_eq!(s.fills[1].date, 3);
        near(s.fills[1].price, level * (1.0 - sign * 0.001));
        let gap = 100.0 * (1.0 + sign * 0.15);
        let s = sim(
            &[bar(0, 100.0), bar(1, 100.0), bar(2, gap)],
            &[target(0, sign * 0.5)],
            free(),
            r,
        );
        assert_eq!(s.fills[1].price, gap);
    }
}

#[test]
fn ambiguous_bar_stops_first_and_adverse_lock_defers_stop() {
    for sign in [-1.0, 1.0] {
        let r = SwingRules {
            stop_loss: Some(0.1),
            take_profit: Some(0.1),
            ..SwingRules::default()
        };
        let s = sim(
            &[
                bar(0, 100.0),
                bar(1, 100.0),
                Bar {
                    high: 120.0,
                    low: 80.0,
                    ..bar(2, 100.0)
                },
            ],
            &[target(0, sign * 0.5)],
            free(),
            r,
        );
        near(
            s.fills[1].price,
            100.0 * (1.0 - sign * 0.1) * (1.0 - sign * 0.001),
        );
        let px = 100.0 * (1.0 - sign * 0.15);
        let locked = if sign > 0.0 {
            Bar {
                limit_down: px,
                ..bar(2, px)
            }
        } else {
            Bar {
                limit_up: px,
                ..bar(2, px)
            }
        };
        let s = sim(
            &[bar(0, 100.0), bar(1, 100.0), locked, bar(3, px)],
            &[target(0, sign * 0.5)],
            free(),
            r,
        );
        assert_eq!(s.fills[1].date, 3);
        assert_eq!(s.fills[1].price, px);
    }
}

#[test]
fn open_entry_can_exit_same_day_but_close_entry_cannot() {
    let b = [
        bar(0, 100.0),
        Bar {
            high: 120.0,
            ..bar(1, 100.0)
        },
    ];
    for execution_close in [false, true] {
        let s = sim(
            &b,
            &[target(0, 0.5)],
            free(),
            SwingRules {
                execution_close,
                take_profit: Some(0.1),
                ..SwingRules::default()
            },
        );
        assert_eq!(s.fills.len(), if execution_close { 1 } else { 2 });
    }
}

#[test]
fn exit_cancels_same_side_pending_but_keeps_reversal_for_close() {
    for sign in [-1.0, 1.0] {
        let b = [
            bar(0, 100.0),
            bar(1, 100.0),
            Bar {
                high: 120.0,
                ..bar(2, 100.0)
            },
            bar(3, 100.0),
        ];
        let s = sim(
            &b,
            &[target(0, 0.5), target(1, sign * 0.6)],
            free(),
            SwingRules {
                execution_close: true,
                take_profit: Some(0.1),
                ..SwingRules::default()
            },
        );
        assert_eq!(s.fills.len(), if sign > 0.0 { 2 } else { 3 });
        if sign < 0.0 {
            assert!(s.fills[2].shares < 0);
            assert_eq!(s.fills[2].date, 2);
        }
    }
}

#[test]
fn slippage_applied_after_limit_test_and_entry_basis_is_weighted() {
    let s = sim(
        &[bar(0, 100.0), bar(1, 100.0), bar(2, 120.0)],
        &[target(0, 0.3), target(1, 0.6)],
        SwingCosts {
            slippage: 0.01,
            ..free()
        },
        SwingRules::default(),
    );
    near(s.fills[0].price, 101.0);
    near(s.fills[1].price, 121.2);
    near(
        s.positions.last().unwrap().entry,
        (3000.0 * 101.0 + 2000.0 * 121.2) / 5000.0,
    );
}

#[test]
fn hedge_follows_actual_book_and_roll_is_charged_twice_after_return() {
    let b = [
        bar(0, 100.0),
        Bar {
            limit_up: 100.0,
            ..bar(1, 100.0)
        },
        bar(2, 100.0),
        bar(3, 110.0),
    ];
    let h = [Hedge {
        date: 3,
        ret: 0.1,
        roll: true,
    }];
    let s = run_swing(
        &b,
        &[target(0, 0.5)],
        1e6,
        &SwingCosts {
            hedge_cost: 0.001,
            ..free()
        },
        &SwingRules {
            hedge_ratio: 1.0,
            ..SwingRules::default()
        },
        None,
        Some(&h),
    )
    .unwrap();
    near(s.daily[1].hedge_notional, 0.0);
    near(s.daily[2].hedge_cost, 500.0);
    near(s.daily[3].hedge_pnl, -50_000.0);
    near(s.daily[3].hedge_cost, 1100.0);
    near(s.daily[3].equity, 998_400.0);
}

#[test]
fn invalid_input_is_rejected_and_empty_input_is_typed() {
    let r = SwingRules::default();
    assert!(run_swing(
        &[bar(0, 100.0), bar(0, 100.0)],
        &[],
        1e6,
        &free(),
        &r,
        None,
        None
    )
    .is_err());
    assert!(run_swing(&[], &[target(0, f64::NAN)], 1e6, &free(), &r, None, None).is_err());
    assert!(run_swing(&[], &[], -1.0, &free(), &r, None, None).is_err());
    assert!(run_swing(
        &[],
        &[],
        1e6,
        &free(),
        &SwingRules {
            hedge_ratio: 1.0,
            ..r
        },
        None,
        None
    )
    .is_err());
    assert!(run_swing(&[], &[], 1e6, &free(), &r, None, None)
        .unwrap()
        .daily
        .is_empty());
}
