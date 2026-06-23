import os

from src import rf_promotion as rfp


def test_should_promote_requires_better_validation_pnl():
    cand = rfp.PolicyMetrics(validation_pnl=1.0, test_pnl=5.0, liquidation_count=1)
    base = rfp.PolicyMetrics(validation_pnl=2.0, test_pnl=1.0, liquidation_count=1)
    ok, reason = rfp.should_promote(cand, base)
    assert ok is False
    assert reason == "validation_pnl_not_better"


def test_should_promote_requires_better_test_pnl():
    cand = rfp.PolicyMetrics(3.0, 1.0, 1)
    base = rfp.PolicyMetrics(2.0, 1.0, 1)
    ok, reason = rfp.should_promote(cand, base)
    assert ok is False
    assert reason == "test_pnl_not_better"


def test_should_promote_blocks_on_excess_liquidations():
    cand = rfp.PolicyMetrics(3.0, 3.0, 5)
    base = rfp.PolicyMetrics(2.0, 2.0, 2)
    ok, reason = rfp.should_promote(cand, base, max_liq_ratio=1.0)
    assert ok is False
    assert reason == "liquidation_count_too_high"


def test_should_promote_success():
    cand = rfp.PolicyMetrics(3.0, 3.0, 1)
    base = rfp.PolicyMetrics(2.0, 2.0, 2)
    ok, reason = rfp.should_promote(cand, base)
    assert ok is True
    assert reason is None


def _make_staging(root, symbol):
    d = os.path.join(root, symbol)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rf_close_decision.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")


def test_promotion_is_per_symbol(tmp_path):
    staging = str(tmp_path / "staging")
    promoted = str(tmp_path / "promoted")
    archive = str(tmp_path / "archive")
    _make_staging(staging, "BTCUSDT")
    _make_staging(staging, "ETHUSDT")

    rfp.promote_symbol("BTCUSDT", staging, promoted, archive)

    assert os.path.isdir(os.path.join(promoted, "BTCUSDT"))
    # ETHUSDT must remain unpromoted.
    assert not os.path.isdir(os.path.join(promoted, "ETHUSDT"))


def test_evaluate_and_promote_blocks_and_reports(tmp_path):
    staging = str(tmp_path / "staging")
    promoted = str(tmp_path / "promoted")
    archive = str(tmp_path / "archive")
    _make_staging(staging, "BTCUSDT")

    cand = rfp.PolicyMetrics(1.0, 1.0, 10)
    base = rfp.PolicyMetrics(2.0, 2.0, 1)
    outcome = rfp.evaluate_and_promote("BTCUSDT", cand, base, staging, promoted, archive)
    assert outcome["model_promoted"] is False
    assert outcome["reason_if_not_promoted"] == "validation_pnl_not_better"
    assert not os.path.isdir(os.path.join(promoted, "BTCUSDT"))
