"""Check the public Python import on 3.9, optionally stubbing legacy native exports.

--stub-native isolates Python compatibility when only the swing binding is built;
it does not verify the full extension ABI or legacy engine.
"""

import dataclasses
import sys
import types


def main():
    original_field = dataclasses.field

    def field_without_kw_only(*args, **kwargs):
        assert "kw_only" not in kwargs, "dataclasses.field kw_only requires Python >=3.10"
        return original_field(*args, **kwargs)

    dataclasses.field = field_without_kw_only
    if "--stub-native" in sys.argv:
        native = types.ModuleType("polars_backtest._polars_backtest")
        for name in [
            "BacktestConfig",
            "BacktestReport",
            "TradeRecord",
            "backtest",
            "backtest_with_report",
            "backtest_wide",
            "backtest_with_report_wide_impl",
        ]:
            setattr(native, name, object)
        native.__version__ = "import-smoke-stub"
        sys.modules[native.__name__] = native

    import polars_backtest as pbt

    costs = pbt.SwingCosts(hedge_cost=0.0001)
    assert costs.hedge_cost == 0.0001
    assert costs.fee_rate == 0.001425
    assert pbt.SwingCosts(0.0001) == costs
    try:
        pbt.SwingCosts()
    except TypeError as exc:
        assert "hedge_cost" in str(exc)
    else:
        raise AssertionError("hedge_cost must remain required")
    print(f"Python {sys.version.split()[0]}: public import and SwingCosts passed")


if __name__ == "__main__":
    main()
