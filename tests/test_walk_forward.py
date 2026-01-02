from trdr.backtest.walk_forward import WalkForwardConfig, generate_folds


def test_generate_folds_requires_min_test_bars_after_warmup():
    wf_config = WalkForwardConfig(n_folds=1, train_pct=0.5, min_test_bars=3)
    folds = generate_folds(total_bars=10, wf_config=wf_config, warmup_bars=2)
    assert folds == []


def test_generate_folds_includes_warmup_in_test_window():
    wf_config = WalkForwardConfig(n_folds=1, train_pct=0.5, min_test_bars=3)
    folds = generate_folds(total_bars=12, wf_config=wf_config, warmup_bars=2)
    assert len(folds) == 1

    fold = folds[0]
    test_size = fold.test_end - fold.test_start
    assert test_size >= 2 + 3
    assert (test_size - 2) >= wf_config.min_test_bars
    assert fold.test_start == fold.train_end
