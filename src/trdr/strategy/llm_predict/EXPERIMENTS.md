# LLM Predict Experiments

This doc summarizes key milestones and exact configs that were run.
Results live in `src/trdr/strategy/llm_predict/results.json`.

Notes

- Runs are not deterministic unless temperature is forced to 0.0.
- All runs use `GeminiPredictor` in single-request mode.
- Data default: `crypto:eth_usd:15min.csv` unless noted.

## Ideas

- DONE: SAX encoding (Gaussian breakpoints) with multi-vote prompts and parameter sweeps.
- DONE: Single-indicator variant (RSI bucket) and HTF RSI bucket.
- DONE: Retrieval-k sweeps and horizon sweeps.
- DONE: Barrier sweeps around 0.15.
- DONE: Training example count sweeps (150, 300, 450).
- DONE: Pattern gate (top-N) hard filter.
- DONE: Returns-based encoding baseline.
- DONE: Gaussian-binned returns (raw/log).
- DONE: SAX momentum arrows (↑/↓/→) instead of U/D/N suffix.
- DONE: SAX rarity bucket token based on pattern frequency (low/med/high).
- DONE: Two-pass label/feature: append potential bucket from max forward return (then use UNK at test).
- TODO: Cross-asset conditioning token (e.g., BTC trend bucket).
- DONE: Retrieval neighbors by SAX similarity (instead of return vector).
- DONE: Predict hold bucket (4/6/8) instead of direction, then map to trade.
- DONE: Looser barrier sweeps (0.25, 0.30) vs baseline 0.15.
- DONE: TOP-pattern token based on forward max-return ranking.
- DONE: Forward max-return potential buckets (avg max-up/max-down per SAX base).
- DONE: Up-only potential bucket token (drop downside bucket).
- DONE: Longer potential horizon for SAX buckets (48/72 bars).
- DONE: Return-profile shape token (future return trajectory SAX in training, UNK at test).
- DONE: Prospect/drawdown buckets (P/D magnitude + time-to-hit, UNK at test).
- DONE: Similarity-to-top-k patterns (TOPHIT:1/0; UNK at test).

- DONE: Training stride increase to reduce near-duplicate examples.
- TODO: Volatility bucket token (ATR% low/med/high).
- TODO: Soft gate token (`PATTERN:UNSEEN`) instead of hard `same`.
- TODO: Cross-asset conditioning token (e.g., BTC trend bucket) with retriever weighting.
- TODO: Time-of-day/session bucket token (UTC hour or session class).

Milestones

1. Baselines (ICL, direction5)

- Algo: single-request ICL with coordinate encoding, direction5 labels.
- Config: `baseline|w=17|t=300|direction5|detailed`
- Result: ~20% overall, ~40% directional

2. Encoding variants (ICL)

- Algo: single-request ICL with alternate encodings (signed/rank/delta bins).
- Signed coordinate: `signed_coord|w=17|t=300|direction5|detailed|enc=signed_coordinate`
- Delta bins: `delta_bins|w=17|t=300|direction5|detailed|enc=delta_bins`
- Rank: `rank|w=17|t=300|direction5|detailed|enc=rank`
- Result: mostly ~40-48% overall, weak directional

3. Retrieval ICL (k-NN)

- Algo: retrieval ICL (k-NN on feature vector), multichannel encoding.
- Multichannel + retrieval: `multi_retrieval|w=17|t=300|direction5|detailed|enc=multichannel,ret=12`
- Result: ~45% overall, ~21% directional

4. Triple-barrier labels (h=6)

- Algo: triple-barrier labels, multichannel encoding, retrieval ICL.
- Base: `triple_barrier_retrieval|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=12,h=6,bu=0.25,bd=0.25`
- Result: ~42.5% overall, ~50% directional

5. Triple-barrier sweep (bu/bd)

- Algo: triple-barrier labels with barrier sweep, multichannel retrieval ICL.
- `triple_barrier_retrieval_020|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=12,h=6,bu=0.2,bd=0.2`
- `triple_barrier_retrieval_030|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=12,h=6,bu=0.3,bd=0.3`
- Result: ~46.7%/51.9% directional for 0.20, ~40%/50% for 0.30

6. Horizon sweep (bu/bd=0.20)

- Algo: triple-barrier labels with horizon sweep, multichannel retrieval ICL.
- `triple_barrier_retrieval_020_h4|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=12,h=4,bu=0.2,bd=0.2`
- `triple_barrier_retrieval_020_h8|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=12,h=8,bu=0.2,bd=0.2`
- Result: h=4 ~45.5% directional, h=8 ~39.3% directional

7. Retrieval-k sweep (bu/bd=0.20, h=6)

- Algo: retrieval ICL neighbor count sweep with triple-barrier labels.
- `triple_barrier_retrieval_020_k8|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=8,h=6,bu=0.2,bd=0.2`
- `triple_barrier_retrieval_020_k16|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- `triple_barrier_retrieval_020_k20|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=20,h=6,bu=0.2,bd=0.2`
- Result: k=16 matched best (~51.9% directional), k=8/20 lower

8. Prompt format sweep (h=6, bu/bd=0.20, k=16)

- Algo: prompt format sweep with triple-barrier labels and retrieval ICL.
- `tb_020_k16_detailed|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- `tb_020_k16_minimal|w=17|t=300|triple_barrier|minimal|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- `tb_020_k16_strict|w=17|t=300|triple_barrier|strict|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- Result: detailed ~63.2% directional (n=19), strict ~57.9%, minimal ~21.1%

9. Multi-vote prompt (verbalized sampling)

- Algo: multi-vote prompt (3 guesses, majority vote), retrieval ICL.
- `tb_020_k16_multivote|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- 20-test run: ~73.7% directional
- 40-test run: ~54.1% directional

10. Temperature=0 (multi-vote)

- Algo: multi-vote prompt with temperature=0 for determinism.
- `tb_020_k16_multivote_temp0|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2,temp=0.0`
- 20-test run: ~68.4% directional
- 40-test run: ~51.4% directional

11. Regime filter (ATR%)

- Algo: regime filter on ATR% with triple-barrier labels and retrieval ICL.
- `tb_020_k16_regime_atr020|w=17|t=300|triple_barrier|detailed|enc=multichannel,ret=16,atr>=0.2,h=6,bu=0.2,bd=0.2`
- Result: ~52.6% directional (n=19)

12. Cross-asset checks

- Algo: apply best ETH config on BTC/AAPL to test generalization.
- BTC: `tb_020_k16_btc|w=17|t=300|triple_barrier|detailed|enc=multichannel,sym=crypto:btc_usd:15min.csv,ret=16,h=6,bu=0.2,bd=0.2`
- AAPL: `tb_020_k16_aapl|w=17|t=300|triple_barrier|detailed|enc=multichannel,sym=stock:aapl:15min.csv,ret=16,h=6,bu=0.2,bd=0.2`
- Result: weak generalization (~25-30% overall)

13. Multi-vote confirmations (40 tests)

- Algo: multi-vote prompt confirmation run.
- `tb_020_k16_multivote_40|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2`
- Result: ~54.1% directional (20/37)

14. Temp=0 confirmations (40 tests)

- Algo: multi-vote prompt confirmation with temperature=0.
- `tb_020_k16_multivote_temp0_40|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,h=6,bu=0.2,bd=0.2,temp=0.0`
- Result: ~51.4% directional (19/37)

15. Regime-filter confirmations

- Algo: multi-vote prompt with ATR filter confirmation.
- `tb_020_k16_multivote_temp0_atr025_30|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,atr>=0.25,h=6,bu=0.2,bd=0.2,temp=0.0`
- Result: ~47.8% directional (11/23)

16. Encoding variants with multi-vote (temp=0, 20 tests)

- Algo: encoding sweep with multi-vote prompt and temperature=0.
- `tb_020_k16_multivote_temp0_h10|w=17|t=300|triple_barrier|multi_vote|enc=multichannel,ret=16,h=10,bu=0.2,bd=0.2,temp=0.0`
- `tb_020_k16_multivote_temp0_signed|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.2,bd=0.2,temp=0.0`
- `tb_020_k16_multivote_temp0_returns|w=17|t=300|triple_barrier|multi_vote|enc=returns,ret=16,h=6,bu=0.2,bd=0.2,temp=0.0`
- Result: small-N boosts (68-74% directional), not confirmed

17. Signed-coordinate confirmation (40 tests)

- Algo: signed-coordinate encoding confirmation with multi-vote and temperature=0.
- `tb_020_k16_multivote_temp0_signed_40|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.2,bd=0.2,temp=0.0`
- Result: ~54.1% directional (20/37)

18. Barrier and horizon sweep (signed-coordinate, temp=0, 20 tests)

- Algo: triple-barrier sweep with signed-coordinate encoding + multi-vote prompt + temp=0.
- `tb_015_k16_multivote_temp0|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- `tb_025_k16_multivote_temp0|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.25,bd=0.25,temp=0.0`
- `tb_020_k16_multivote_temp0_h8|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=8,bu=0.2,bd=0.2,temp=0.0`
- `tb_020_k16_multivote_temp0_h12|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=12,bu=0.2,bd=0.2,temp=0.0`
- Result: small-N boosts (68-75% directional), not confirmed

19. Barrier confirmation (signed-coordinate, temp=0, 40 tests)

- Algo: confirm barrier size with signed-coordinate encoding + multi-vote prompt + temp=0.
- `tb_015_k16_multivote_temp0_40|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- `tb_025_k16_multivote_temp0_40|w=17|t=300|triple_barrier|multi_vote|enc=signed_coordinate,ret=16,h=6,bu=0.25,bd=0.25,temp=0.0`
- Result: bu/bd=0.15 ~55.0% directional (22/40), bu/bd=0.25 ~53.1% directional (17/32)

20. Multi-vote5 short run (signed-coordinate, temp=0, 20 tests)

- Algo: 5-guess verbalized sampling with signed-coordinate encoding + temp=0.
- `tb_015_k16_multivote5_temp0_20|w=17|t=300|triple_barrier|multi_vote5|enc=signed_coordinate,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~70.0% directional (14/20), not confirmed

21. Multi-vote5 confirmation (signed-coordinate, temp=0, 40 tests)

- Algo: 5-guess verbalized sampling confirmation run.
- `tb_015_k16_multivote5_temp0_40|w=17|t=300|triple_barrier|multi_vote5|enc=signed_coordinate,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (22/40)

22. SAX encoding short run (multi-vote, temp=0, 20 tests)

- Algo: SAX encoding with Gaussian breakpoints + momentum/range flags, multi-vote prompt.
- `tb_015_k16_multivote_temp0_sax_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=4,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), not confirmed

23. SAX encoding confirmation (multi-vote, temp=0, 40 tests)

- Algo: SAX encoding confirmation run with Gaussian breakpoints + momentum/range flags.
- `tb_015_k16_multivote_temp0_sax_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=4,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40)

24. SAX + RSI (single indicator) short run (multi-vote, temp=0, 20 tests)

- Algo: SAX encoding with a single RSI bucket (no extra indicators), multi-vote prompt.
- `tb_015_k16_multivote_temp0_sax_rsi_20|w=17|t=300|triple_barrier|multi_vote|enc=sax_rsi,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (13/20), not confirmed

25. SAX + RSI + HTF RSI (4h) short run (multi-vote, temp=0, 20 tests)

- Algo: SAX + RSI bucket with higher-timeframe RSI bucket appended.
- `tb_015_k16_multivote_temp0_sax_rsi_htf4h_20|w=17|t=300|triple_barrier|multi_vote|enc=sax_rsi,htf=crypto:eth_usd:4hour.csv,htf_rsi=14,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~70.0% directional (14/20), not confirmed

26. SAX 4h baseline (multi-vote, temp=0, 20 tests)

- Algo: SAX encoding on 4h bars with multi-vote prompt.
- `tb_015_k16_multivote_temp0_sax_4h_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:4hour.csv,sax_p=5,sax_a=4,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~50.0% directional (10/20), not confirmed

27. SAX parameter sweep (multi-vote, temp=0, 20 tests)

- Algo: SAX parameter sweep (paa/alphabet) with multi-vote prompt.
- `tb_015_k16_multivote_temp0_sax_p4_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=4,sax_a=4,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- `tb_015_k16_multivote_temp0_sax_p6_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=6,sax_a=4,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- `tb_015_k16_multivote_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: p4 ~70.0%, p6 ~70.0%, a5 ~75.0% directional (small-N)

28. SAX a=5 confirmation (multi-vote, temp=0, 40 tests)

- Algo: SAX encoding confirmation with alphabet size 5.
- `tb_015_k16_multivote_temp0_sax_a5_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~67.5% directional (27/40)

29. Retrieval-k sweep on SAX a=5 (multi-vote, temp=0, 20 tests)

- Algo: retrieval neighbor count sweep on SAX a=5 baseline.
- `tb_015_k8_multivote_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=8,h=6,bu=0.15,bd=0.15,temp=0.0`
- `tb_015_k24_multivote_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=24,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: k=8 ~55.0% directional (11/20), k=24 ~75.0% directional (15/20)

30. SAX flag permutations + RSI toggle (multi-vote, temp=0, 20 tests)

- Algo: ablations over SAX flags (M/R/E) with optional RSI bucket.
- Best short-run: `tb_015_k16_multivote_temp0_sax_a5_m0r0e0rsi1_20` at ~85.0% directional (17/20)
- Also strong: `m1r0e0rsi0`, `m0r0e1rsi0`, `m1r1e0rsi0` at ~80.0% directional
- All permutations logged in `results.json` (20-test runs, not confirmed).

31. SAX RSI-only confirmation (multi-vote, temp=0, 40 tests)

- Algo: SAX a=5 with RSI bucket only (no M/R/E flags), confirmation run.
- `tb_015_k16_multivote_temp0_sax_a5_m0r0e0rsi1_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_m=0,sax_r=0,sax_e=0,sax_rsi=14,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~62.5% directional (25/40)

32. Retrieval-k confirmation (k=24, SAX a=5, 40 tests)

- Algo: confirm higher retrieval_k for SAX a=5.
- `tb_015_k24_multivote_temp0_sax_a5_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=24,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (22/40)

33. Barrier sweep around 0.15 (SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: tighten/loosen barrier around 0.15 with SAX a=5 baseline.
- `tb_012_k16_multivote_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.12,bd=0.12,temp=0.0`
- `tb_018_k16_multivote_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.18,bd=0.18,temp=0.0`
- Result: 0.12 ~65.0% directional (13/20), 0.18 ~50.0% directional (10/20)

34. Horizon sweep (SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: shorter horizon on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_h4_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=4,bu=0.15,bd=0.15,temp=0.0`
- Result: h=4 ~73.7% directional (14/19), not confirmed

35. Horizon confirmation (h=4, SAX a=5, multi-vote, temp=0, 40 tests)

- Algo: h=4 confirmation on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_h4_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=4,bu=0.15,bd=0.15,temp=0.0`
- Result: ~67.6% directional (25/37)

36. Horizon sweep (h=8, SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: longer horizon on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_h8_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=8,bu=0.15,bd=0.15,temp=0.0`
- Result: h=8 ~70.0% directional (14/20), not confirmed

37. SAX a=5 with multi-vote5 (temp=0, 20 tests)

- Algo: 5-guess verbalized sampling on SAX a=5 baseline.
- `tb_015_k16_multivote5_temp0_sax_a5_20|w=17|t=300|triple_barrier|multi_vote5|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), not confirmed

38. SAX a=5 with multi-vote5 confirmation (temp=0, 40 tests)

- Algo: 5-guess verbalized sampling confirmation on SAX a=5 baseline.
- `tb_015_k16_multivote5_temp0_sax_a5_40|w=17|t=300|triple_barrier|multi_vote5|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (26/40)

39. Training examples increase (SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: increase training examples to 450 on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_450|w=17|t=450|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~45.0% directional (9/20), worse

40. Training examples decrease (SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: reduce training examples to 150 on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_150|w=17|t=150|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~50.0% directional (10/20)

41. SAX pattern gate (top 50) short run (multi-vote, temp=0, 20 tests)

- Algo: only call LLM when SAX pattern is in top-50 training patterns; otherwise predict `same`.
- `tb_015_k16_multivote_temp0_sax_a5_gate50_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_gate=50,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~20.0% directional (4/20), very poor

42. Horizon confirmation (h=8, SAX a=5, multi-vote, temp=0, 40 tests)

- Algo: h=8 confirmation on SAX a=5 baseline.
- `tb_015_k16_multivote_temp0_sax_a5_h8_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=8,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (26/40)

43. SAX a=6 short run (multi-vote, temp=0, 20 tests)

- Algo: SAX alphabet size 6 on baseline config.
- `tb_015_k16_multivote_temp0_sax_a6_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=6,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (13/20)

44. Returns encoding baseline (multi-vote, temp=0, 20 tests)

- Algo: returns encoding with same label/horizon/retrieval settings.
- `tb_015_k16_multivote_temp0_returns_20|w=17|t=300|triple_barrier|multi_vote|enc=returns,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (12/20)

45. Log-returns encoding baseline (multi-vote, temp=0, 20 tests)

- Algo: log-returns encoding with same label/horizon/retrieval settings.
- `tb_015_k16_multivote_temp0_log_returns_20|w=17|t=300|triple_barrier|multi_vote|enc=log_returns,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~80.0% directional (16/20), not confirmed

46. SAX on log-returns (multi-vote, temp=0, 20 tests)

- Algo: SAX encoding applied to log-returns (Gaussian breakpoints).
- `tb_015_k16_multivote_temp0_sax_lr_a5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax_log_returns,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (12/20)

47. Log-returns confirmation (multi-vote, temp=0, 40 tests)

- Algo: log-returns encoding confirmation run.
- `tb_015_k16_multivote_temp0_log_returns_40|w=17|t=300|triple_barrier|multi_vote|enc=log_returns,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40)

48. SAX occurrence bucket (multi-vote, temp=0, 20 tests)

- Algo: add `OCC:L/M/H` based on pattern frequency in training.
- `tb_015_k16_multivote_temp0_sax_a5_occ_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_occ=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (11/20)

49. Training stride (SAX a=5, multi-vote, temp=0, 20 tests)

- Algo: increase training sampling stride to reduce overlap.
- `tb_015_k16_multivote_temp0_sax_a5_stride5_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,stride=5,temp=0.0`
- Result: ~65.0% directional (13/20)

50. Training stride confirmation (SAX a=5, multi-vote, temp=0, 40 tests)

- Algo: confirm stride=5 over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_stride5_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,stride=5,temp=0.0`
- Result: ~55.0% directional (22/40), worse than baseline.

51. SAX arrows for momentum (multi-vote, temp=0, 20 tests)

- Algo: same as best SAX, but momentum suffix uses arrows (↑/↓/→) instead of U/D/N.
- `tb_015_k16_multivote_temp0_sax_a5_arrows_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_arrows=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (13/20), below best.

52. SAX triangles for momentum (multi-vote, temp=0, 20 tests)

- Algo: same as best SAX, but momentum suffix uses symbols (▲/▼/■) instead of U/D/N.
- `tb_015_k16_multivote_temp0_sax_a5_tri_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_tri=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), promising but needs confirm.

53. SAX range symbols (multi-vote, temp=0, 20 tests)

- Algo: same as best SAX, but range suffix uses symbols (▮/━/□) instead of F/W/N.
- `tb_015_k16_multivote_temp0_sax_a5_range_sym_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_range_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), promising but needs confirm.

54. SAX all symbols (multi-vote, temp=0, 20 tests)

- Algo: use symbols for momentum (▲/▼/■), range (▮/━/□), and RSI (▁/▄/█).
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~80.0% directional (16/20), promising but needs confirm.

55. SAX all symbols + patch summary (multi-vote, temp=0, 20 tests)

- Algo: add patch summary tokens to the all-symbols SAX variant.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_patch_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,patch=4,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (13/20), worse than base.

56. SAX all symbols confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm all-symbols SAX variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~67.5% directional (27/40), matches baseline.

57. Gaussian-binned returns (multi-vote, temp=0, 20 tests)

- Algo: map per-bar returns into Gaussian bins (no PAA), using digits 0..4.
- `tb_015_k16_multivote_temp0_gauss_ret_20|w=17|t=300|triple_barrier|multi_vote|enc=gauss_returns,gauss_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~70.0% directional (14/20), needs confirm.

58. Gaussian-binned log returns (multi-vote, temp=0, 20 tests)

- Algo: map per-bar log returns into Gaussian bins (no PAA), using digits 0..4.
- `tb_015_k16_multivote_temp0_gauss_lr_20|w=17|t=300|triple_barrier|multi_vote|enc=gauss_log_returns,gauss_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~70.0% directional (14/20), needs confirm.

59. Gaussian-binned log returns confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm gauss log-returns over 40 tests.
- `tb_015_k16_multivote_temp0_gauss_lr_40|w=17|t=300|triple_barrier|multi_vote|enc=gauss_log_returns,gauss_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~52.5% directional (21/40), not promising.

60. Gaussian-binned returns confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm gauss returns over 40 tests.
- `tb_015_k16_multivote_temp0_gauss_ret_40|w=17|t=300|triple_barrier|multi_vote|enc=gauss_returns,gauss_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~52.5% directional (21/40), not promising.

61. All-symbols SAX on BTC (multi-vote, temp=0, 20 tests)

- Algo: same as best all-symbols SAX, run on BTC 15m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_btc_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:btc_usd:15min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~40.0% directional (8/20), weak.

62. All-symbols SAX on AAPL (multi-vote, temp=0, 20 tests)

- Algo: same as best all-symbols SAX, run on AAPL 15m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_aapl_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=stock:aapl:15min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~35.0% directional (7/20), weak.

63. SAX rarity bucket (multi-vote, temp=0, 20 tests)

- Algo: add `OCC:L/M/H` based on pattern frequency in training.
- `tb_015_k16_multivote_temp0_sax_a5_occ_20_v2|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_occ=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (12/20), weak.

64. SAX similarity retrieval (multi-vote, temp=0, 20 tests)

- Algo: retrieval neighbors selected by SAX base Hamming distance (not feature vector).
- `tb_015_k16_multivote_temp0_sax_a5_saxret_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,sax_ret=1,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~40.0% directional (8/20), weak.

65. All-symbols SAX confirmation (multi-vote, temp=0, 40 tests, v2)

- Algo: re-confirm all-symbols SAX variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_40_v2|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~67.5% directional (27/40), matches baseline.

66. SAX triangles confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm momentum triangles (▲/▼/■) over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_tri_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_tri=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40), not promising.

67. SAX range symbols confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm range-symbols variant (▮/━/□) over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_range_sym_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_range_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~62.5% directional (25/40), not better.

68. Hold-bucket labels (multi-vote, temp=0, 20 tests)

- Algo: predict best hold bucket (H4/H6/H8) by max absolute return; directional score is long-win rate at predicted hold.
- `tb_015_k16_multivote_temp0_sax_a5_holdbucket_20|w=17|t=300|hold_bucket|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,temp=0.0`
- Result: ~30.0% overall (6/20), ~40.0% directional (8/20), weak.

69. Base SAX reconfirm (multi-vote, temp=0, 40 tests, v2)

- Algo: reconfirm base SAX setup.
- `tb_015_k16_multivote_temp0_sax_a5_40_v2|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (24/40), below prior confirms.

70. All-symbols SAX reconfirm (multi-vote, temp=0, 40 tests, v3)

- Algo: reconfirm all-symbols SAX setup.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_40_v3|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~67.5% directional (27/40), matches prior confirms.

71. h=4 reconfirm (multi-vote, temp=0, 40 tests, v2)

- Algo: reconfirm horizon=4 variant.
- `tb_015_k16_multivote_temp0_sax_a5_h4_40_v2|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,ret=16,h=4,bu=0.15,bd=0.15,temp=0.0`
- Result: ~62.2% directional (23/37), weaker than prior.

72. All-symbols SAX without momentum (multi-vote, temp=0, 20 tests)

- Algo: all-symbols SAX with momentum disabled (no `M` flag).
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_m0_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_m=0,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (11/20), worse.

73. All-symbols SAX on ETH 5m (multi-vote, temp=0, 20 tests)

- Algo: same as all-symbols SAX, run on ETH 5m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth5m_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:5min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~35.0% overall (7/20), ~53.8% directional (7/13), weak.

74. All-symbols SAX on ETH 7m (multi-vote, temp=0, 20 tests)

- Algo: same as all-symbols SAX, run on ETH 7m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth7m_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:7min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (11/20), weak.

75. All-symbols SAX on ETH 11m (multi-vote, temp=0, 20 tests)

- Algo: same as all-symbols SAX, run on ETH 11m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth11m_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:11min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% overall (12/20), ~70.6% directional (12/17), promising but needs confirm.

76. All-symbols SAX on ETH 11m confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm 11m variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth11m_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:11min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% overall (22/40), ~62.9% directional (22/35), weaker.

77. All-symbols SAX on ETH 3m (multi-vote, temp=0, 20 tests)

- Algo: same as all-symbols SAX, run on ETH 3m.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth3m_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:3min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~20.0% overall (4/20), ~30.8% directional (4/13), very weak.

78. All-symbols SAX on ETH 11m reconfirm (multi-vote, temp=0, 40 tests, v2)

- Algo: reconfirm 11m variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_eth11m_40_v2|w=17|t=300|triple_barrier|multi_vote|enc=sax,sym=crypto:eth_usd:11min.csv,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% overall (22/40), ~62.9% directional (22/35), consistent with prior confirm.

79. All-symbols SAX without ordinal prefix (multi-vote, temp=0, 20 tests)

- Algo: remove the `#<n>` ordinal prefix from training examples.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_noidx_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), promising but needs confirm.

80. All-symbols SAX without ordinal prefix confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm no-ordinal variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_noidx_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (24/40), worse than baseline.

81. All-symbols SAX without # prefix (multi-vote, temp=0, 20 tests)

- Algo: keep ordinal numbers but remove the `#` prefix.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_nohash_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), promising but needs confirm.

82. All-symbols SAX without # prefix confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm no-# variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_nohash_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~62.5% directional (25/40), worse than baseline.

83. All-symbols SAX, bu/bd=0.30 (multi-vote, temp=0, 20 tests)

- Algo: loosen barriers to 0.30 (h=6).
- `tb_030_k16_multivote_temp0_sax_a5_symbols_all_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.3,bd=0.3,temp=0.0`
- Result: ~71.4% directional (10/14), overall 50.0% (10/20); promising directional but mixed overall.

84. All-symbols SAX, bu/bd=0.30 confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm 0.30 barrier variant over 40 tests.
- `tb_030_k16_multivote_temp0_sax_a5_symbols_all_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.3,bd=0.3,temp=0.0`
- Result: ~50.0% directional (15/30), overall 37.5% (15/40), weak.

85. Fixed-hold labels (h=8) (multi-vote, temp=0, 20 tests)

- Algo: label UP/DOWN by fixed-hold return (h=8), ignore barriers.
- `fh_8_k16_multivote_temp0_sax_a5_symbols_all_20|w=17|t=300|fixed_hold|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,temp=0.0`
- Result: ~60.0% directional (12/20), not better.

86. Trailing-barrier labels (multi-vote, temp=0, 20 tests)

- Algo: label DOWN on first down-barrier hit; if up-barrier hits, hold to horizon and score final; else same.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_trail_20|w=17|t=300|trailing_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~65.0% directional (13/20), not better.

87. No retrieval examples (multi-vote, temp=0, 20 tests)

- Algo: use retrieval prompt style but omit the training examples (prompt-only).
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_noretex_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,ret_ex=0,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~70.0% directional (14/20).

88. No retrieval examples confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm prompt-only variant over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_noretex_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,ret_ex=0,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~55.0% directional (22/40), not better.

89. TOP-pattern token (multi-vote, temp=0, 20 tests)

- Algo: append `|TOP:1/0` to training examples based on top-50 SAX bases by avg max-forward return (24 bars); test uses `TOP:?`.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_top50_20|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_top=50,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~75.0% directional (15/20), promising but needs confirm.

90. TOP-pattern token confirmation (multi-vote, temp=0, 40 tests)

- Algo: confirm TOP-pattern token over 40 tests.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_top50_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_top=50,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (24/40), worse than baseline.

91. SAX potential buckets (multi-vote, temp=0, 40 tests)

- Algo: append `|P:U{0..2}D{0..2}` based on avg max-up/max-down over 24 bars for each SAX base (3 buckets).
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_potb3_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_potb=3,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40), not better.

92. SAX potential buckets, longer horizon (multi-vote, temp=0, 40 tests)

- Algo: same as #91 but potential horizon = 48 bars.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_potb3_h48_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_potb=3,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40), not better.

93. SAX potential buckets, upside only (multi-vote, temp=0, 40 tests)

- Algo: same as #91 but keep only upside bucket token (`|P:U0..2`), drop downside.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_potb3_uonly_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_potb=3,sax_potu=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~52.5% directional (21/40), worse.

94. SAX potential buckets, horizon=72 (multi-vote, temp=0, 40 tests)

- Algo: same as #91 but potential horizon = 72 bars.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_potb3_h72_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_potb=3,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~50.0% directional (20/40), worse.

95. Return-profile shape token (multi-vote, temp=0, 40 tests)

- Algo: append `|FUT:<SAX>` for future return trajectory (12 bars) in training; test uses `FUT:?`.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_futshape12_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_fut=12,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (24/40), not better.

96. Prospect/drawdown buckets (multi-vote, temp=0, 40 tests)

- Algo: append `|PROS:U{0..2}D{0..2}` based on max-up/max-down over 12 bars; test uses `PROS:U?D?`.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_prospect12_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_prosp=12:3,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~60.0% directional (24/40), not better.

97. Prospect/drawdown time buckets (multi-vote, temp=0, 40 tests)

- Algo: append `|PT:TU{0..2}TD{0..2}` based on time-to-max/time-to-min over 12 bars; test uses `PT:TU?TD?`.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_prospect12_time_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,sax_prosp=12:3,sax_prosp_t=1,ret=16,h=6,bu=0.15,bd=0.15,temp=0.0`
- Result: ~57.5% directional (23/40), not better.

98. Training stride=5 (multi-vote, temp=0, 40 tests)

- Algo: reduce near-duplicate training examples by using stride=5.
- `tb_015_k16_multivote_temp0_sax_a5_symbols_all_stride5_40|w=17|t=300|triple_barrier|multi_vote|enc=sax,sax_p=5,sax_a=5,sax_rsi=14,sax_tri=1,sax_range_symbols=1,sax_rsi_symbols=1,ret=16,h=6,bu=0.15,bd=0.15,stride=5,temp=0.0`
- Result: ~65.0% directional (26/40), slightly below baseline.

99. Ensemble (base + stride5) (multi-vote, temp=0, 40 tests)

- Algo: run base SAX and stride5+symbols configs in parallel; if they disagree, pick base prediction.
- `ensemble_base+stride5_40`
- Result: ~60.0% directional (24/40), not better.

## Current best (ETH only)

- Best confirmed: `tb_015_k16_multivote_temp0_sax_a5_40` at ~67.5% directional (27/40).
- Runner-up confirmed: `tb_015_k16_multivote_temp0_sax_a5_h4_40` at ~67.6% directional (25/37) but reconfirm fell to ~62.2%.
- Other confirmed: `tb_015_k16_multivote_temp0_sax_a5_symbols_all_stride5_40` at ~65.0% directional (26/40).
- Best unconfirmed (20 tests): `tb_015_k16_multivote_temp0_sax_a5_tri_20` and `tb_015_k16_multivote_temp0_sax_a5_range_sym_20` at ~75.0% directional (15/20) but both fell to 57.5%/62.5% on 40-test confirms.

## Recommended path

- Use the confirmed base: `tb_015_k16_multivote_temp0_sax_a5_40` (67.5% directional).
- If needed, try horizon=4 variant: `tb_015_k16_multivote_temp0_sax_a5_h4_40` (67.6% directional).
- Discard symbol swaps (triangles, range, arrows), rarity buckets, SAX-retrieval, Gaussian bins, and hold-bucket labels—they regressed on 40-test confirms.
- Other assets (BTC, AAPL) underperform markedly with current recipe; treat ETH 15m as the only viable setup so far.
