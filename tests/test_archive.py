from pathlib import Path

from trdr.storage.archive import RunArchive


def test_run_id_sanitizes_symbol(tmp_path: Path) -> None:
    """Run ID uses lowercase symbol with / replaced by _."""
    archive = RunArchive(tmp_path)
    run_id = archive.start_run(symbol="crypto:BTC/USD", config={})
    assert "/" not in run_id
    assert "crypto:btc_usd" in run_id
