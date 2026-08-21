# CPU E20 Post-processing Sweep — Provisional

**Date:** 2026-08-21 (Asia/Dubai)  
**Checkpoint:** `runs/cpu_filanet_20epoch/best.pt`  
**Command:** `python scripts/tune_postprocess.py --data-dir data --checkpoint runs/cpu_filanet_20epoch/best.pt --limit 60`

## Status

This result is **provisional**. The current `tune_postprocess.py` selects the final 60 annotation records with `indices = list(range(total - n, total))`; this is not guaranteed to reproduce the grouped held-out split used during training. The sweep is retained because it is useful for parameter-response analysis, but it must not be presented as the final held-out estimate.

## Inference cost

- Records: 60 of 1154 annotation records
- Probability-map inference time: 3959 s (~66.0 min)
- Configurations swept: 144
- Probability maps were cached for reuse.

## Best configuration reported by the sweep

| parameter / metric | value |
|---|---:|
| threshold | 0.93 |
| min confidence | 0.00 |
| merge gap | 40.0 |
| min-area fraction | 1.2e-4 |
| matched Dice | 0.3005 |
| matched Dice over truth | 0.5308 |
| mean paired Dice | 0.6629 |
| foreground Dice | 0.5160 |
| PQ | 0.2223 |
| RQ | 0.3549 |
| mean instances / observation | 10.45 |
| spurious | 270 |
| one-to-many | 32 |
| missed | 78 |

## Comparison with the corresponding gap=18 configuration

At the same threshold (0.93), confidence (0), and min-area fraction (1.2e-4):

| merge gap | matched Dice | foreground Dice | PQ | RQ | mean instances | spurious | one-to-many | missed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 0.2918 | 0.5160 | 0.2222 | 0.3540 | 11.8 | 316 | 37 | 78 |
| **40** | **0.3005** | **0.5160** | **0.2223** | **0.3549** | **10.45** | **270** | **32** | **78** |

The matched-Dice improvement from merge gap 18 to 40 is approximately **3.0% relative**, while foreground Dice is unchanged. This supports the diagnosis that instance construction matters independently of union-mask overlap.

## Main empirical trends

1. The best region remains at a high probability threshold (`0.93`) with the smallest tested area floor (`1.2e-4`).
2. Increasing `merge_gap` from 18 to 40 helps at threshold 0.93 by reducing fragment count/spurious predictions without changing foreground Dice.
3. `merge_gap=72` is generally worse than 18–40.
4. `min_area_fraction=1e-3` is too aggressive: spurious detections fall, but missed filaments rise sharply and matched Dice collapses.
5. Confidence settings below or equal to the pixel threshold are often redundant. In particular, at threshold 0.93 all tested confidence values (0, 0.60, 0.75, 0.85) produce identical results because the instance pixels already satisfy probability >= 0.93.
6. The best matched Dice (0.3005) remains far below `mean_paired_dice=0.6629`, showing that detection/counting errors remain a major limitation even when matched boundaries are substantially better.

## Next required validation

Before generating another Kaggle submission from this sweep:

- reconstruct the exact grouped validation split used by training;
- run a narrowed post-processing grid on that true held-out subset;
- use cache keys tied to checkpoint, tile size, TTA, and split;
- refine threshold around 0.90–0.96 and merge gap around 25–50;
- test confidence only at values meaningfully above the selected pixel threshold.

## Full raw sweep output

```text
tuning on 60 held-out observations of 1154

  inference 10/60  (679s)
  inference 20/60  (1349s)
  inference 30/60  (2026s)
  inference 40/60  (2699s)
  inference 50/60  (3329s)
  inference 60/60  (3959s)

sweeping 144 configurations over the cached maps
   thr  conf    gap  minarea  matchedD  fgDice     PQ     RQ   inst   spur  1->m  miss
  0.50  0.00   18.0  1.2e-04    0.1947  0.3742 0.0577 0.0989   16.5    582    34    35
  0.50  0.00   18.0  4.0e-04    0.2320  0.3909 0.0702 0.1201   11.9    339    30    61
  0.50  0.00   18.0  1.0e-03    0.2041  0.3689 0.0739 0.1252    5.2     95     9   200
  0.50  0.00   40.0  1.2e-04    0.2008  0.3742 0.0530 0.0906   13.7    448    26    35
  0.50  0.00   40.0  4.0e-04    0.2352  0.3909 0.0686 0.1171   10.2    271    24    61
  0.50  0.00   40.0  1.0e-03    0.2037  0.3689 0.0748 0.1265    5.0     87     8   200
  0.50  0.00   72.0  1.2e-04    0.1902  0.3742 0.0482 0.0820   10.7    321    17    35
  0.50  0.00   72.0  4.0e-04    0.2202  0.3909 0.0599 0.1014    8.6    209    17    61
  0.50  0.00   72.0  1.0e-03    0.1906  0.3689 0.0622 0.1042    4.5     73     5   200
  0.50  0.60   18.0  1.2e-04    0.1972  0.3761 0.0579 0.0990   15.9    551    34    39
  0.50  0.60   18.0  4.0e-04    0.2323  0.3909 0.0699 0.1192   11.8    336    30    62
  0.50  0.60   18.0  1.0e-03    0.2041  0.3689 0.0739 0.1252    5.2     95     9   200
  0.50  0.60   40.0  1.2e-04    0.2041  0.3756 0.0538 0.0915   13.2    423    26    36
  0.50  0.60   40.0  4.0e-04    0.2357  0.3909 0.0685 0.1165   10.2    268    24    62
  0.50  0.60   40.0  1.0e-03    0.2037  0.3689 0.0748 0.1265    5.0     87     8   200
  0.50  0.60   72.0  1.2e-04    0.1925  0.3751 0.0486 0.0823   10.4    303    17    36
  0.50  0.60   72.0  4.0e-04    0.2212  0.3909 0.0600 0.1011    8.5    206    17    62
  0.50  0.60   72.0  1.0e-03    0.1906  0.3689 0.0622 0.1042    4.5     73     5   200
  0.50  0.75   18.0  1.2e-04    0.2221  0.3862 0.0660 0.1128   12.5    371    29    55
  0.50  0.75   18.0  4.0e-04    0.2406  0.3941 0.0731 0.1251   10.6    270    28    71
  0.50  0.75   18.0  1.0e-03    0.2062  0.3741 0.0749 0.1269    5.0     85     9   205
  0.50  0.75   40.0  1.2e-04    0.2259  0.3833 0.0612 0.1042   10.7    289    23    51
  0.50  0.75   40.0  4.0e-04    0.2425  0.3921 0.0708 0.1208    9.2    221    22    71
  0.50  0.75   40.0  1.0e-03    0.2051  0.3733 0.0757 0.1281    4.9     80     8   205
  0.50  0.75   72.0  1.2e-04    0.2131  0.3843 0.0559 0.0945    8.4    201    14    50
  0.50  0.75   72.0  4.0e-04    0.2248  0.3913 0.0625 0.1056    7.8    173    15    71
  0.50  0.75   72.0  1.0e-03    0.1914  0.3733 0.0628 0.1052    4.3     66     5   205
  0.50  0.85   18.0  1.2e-04    0.2009  0.3760 0.0764 0.1304    3.9     58     7   253
  0.50  0.85   18.0  4.0e-04    0.2003  0.3784 0.0773 0.1321    3.6     46     7   256
  0.50  0.85   18.0  1.0e-03    0.1633  0.3358 0.0741 0.1255    2.4     24     3   308
  0.50  0.85   40.0  1.2e-04    0.1856  0.3579 0.0697 0.1184    3.3     48     4   272
  0.50  0.85   40.0  4.0e-04    0.1843  0.3567 0.0696 0.1188    3.2     42     4   272
  0.50  0.85   40.0  1.0e-03    0.1619  0.3345 0.0746 0.1262    2.3     23     2   309
  0.50  0.85   72.0  1.2e-04    0.1525  0.3362 0.0613 0.1028    2.5     32     2   288
  0.50  0.85   72.0  4.0e-04    0.1586  0.3463 0.0575 0.0965    2.6     30     2   281
  0.50  0.85   72.0  1.0e-03    0.1483  0.3246 0.0603 0.1005    2.1     18     0   306
  0.70  0.00   18.0  1.2e-04    0.2264  0.4308 0.1101 0.1861   15.3    514    29    43
  0.70  0.00   18.0  4.0e-04    0.2729  0.4555 0.1308 0.2185    9.5    234    20   100
  0.70  0.00   18.0  1.0e-03    0.1799  0.3748 0.1041 0.1675    3.0     53     5   301
  0.70  0.00   40.0  1.2e-04    0.2298  0.4308 0.1062 0.1798   12.8    402    23    43
  0.70  0.00   40.0  4.0e-04    0.2728  0.4555 0.1311 0.2191    8.5    197    15   100
  0.70  0.00   40.0  1.0e-03    0.1777  0.3748 0.1044 0.1682    2.9     51     4   301
  0.70  0.00   72.0  1.2e-04    0.2048  0.4308 0.0877 0.1484   10.2    308    16    44
  0.70  0.00   72.0  4.0e-04    0.2511  0.4555 0.1061 0.1781    7.3    162    11   100
  0.70  0.00   72.0  1.0e-03    0.1663  0.3748 0.0876 0.1418    2.6     43     4   301
  0.70  0.60   18.0  1.2e-04    0.2264  0.4308 0.1101 0.1861   15.3    514    29    43
  0.70  0.60   18.0  4.0e-04    0.2729  0.4555 0.1308 0.2185    9.5    234    20   100
  0.70  0.60   18.0  1.0e-03    0.1799  0.3748 0.1041 0.1675    3.0     53     5   301
  0.70  0.60   40.0  1.2e-04    0.2298  0.4308 0.1062 0.1798   12.8    402    23    43
  0.70  0.60   40.0  4.0e-04    0.2728  0.4555 0.1311 0.2191    8.5    197    15   100
  0.70  0.60   40.0  1.0e-03    0.1777  0.3748 0.1044 0.1682    2.9     51     4   301
  0.70  0.60   72.0  1.2e-04    0.2048  0.4308 0.0877 0.1484   10.2    308    16    44
  0.70  0.60   72.0  4.0e-04    0.2511  0.4555 0.1061 0.1781    7.3    162    11   100
  0.70  0.60   72.0  1.0e-03    0.1663  0.3748 0.0876 0.1418    2.6     43     4   301
  0.70  0.75   18.0  1.2e-04    0.2265  0.4309 0.1101 0.1861   15.2    513    29    43
  0.70  0.75   18.0  4.0e-04    0.2729  0.4555 0.1308 0.2185    9.5    234    20   100
  0.70  0.75   18.0  1.0e-03    0.1799  0.3748 0.1041 0.1675    3.0     53     5   301
  0.70  0.75   40.0  1.2e-04    0.2299  0.4309 0.1063 0.1798   12.8    401    23    43
  0.70  0.75   40.0  4.0e-04    0.2728  0.4555 0.1311 0.2191    8.5    197    15   100
  0.70  0.75   40.0  1.0e-03    0.1777  0.3748 0.1044 0.1682    2.9     51     4   301
  0.70  0.75   72.0  1.2e-04    0.2049  0.4309 0.0878 0.1485   10.1    307    16    44
  0.70  0.75   72.0  4.0e-04    0.2511  0.4555 0.1061 0.1781    7.3    162    11   100
  0.70  0.75   72.0  1.0e-03    0.1663  0.3748 0.0876 0.1418    2.6     43     4   301
  0.70  0.85   18.0  1.2e-04    0.2523  0.4423 0.1225 0.2070   12.9    391    25    59
  0.70  0.85   18.0  4.0e-04    0.2757  0.4572 0.1330 0.2223    9.1    216    20   107
  0.70  0.85   18.0  1.0e-03    0.1804  0.3755 0.1041 0.1675    3.0     51     5   301
  0.70  0.85   40.0  1.2e-04    0.2554  0.4409 0.1172 0.1983   10.9    305    21    57
  0.70  0.85   40.0  4.0e-04    0.2748  0.4564 0.1331 0.2224    8.1    183    15   107
  0.70  0.85   40.0  1.0e-03    0.1781  0.3755 0.1044 0.1682    2.9     49     4   301
  0.70  0.85   72.0  1.2e-04    0.2264  0.4401 0.0967 0.1635    8.7    233    14    58
  0.70  0.85   72.0  4.0e-04    0.2528  0.4563 0.1077 0.1807    7.0    149    11   107
  0.70  0.85   72.0  1.0e-03    0.1668  0.3755 0.0876 0.1418    2.6     41     4   301
  0.85  0.00   18.0  1.2e-04    0.2651  0.4864 0.1819 0.2972   14.1    438    29    52
  0.85  0.00   18.0  4.0e-04    0.2813  0.4712 0.1905 0.3091    6.6    132    12   179
  0.85  0.00   18.0  1.0e-03    0.1311  0.3068 0.1052 0.1616    1.7     30     1   361
  0.85  0.00   40.0  1.2e-04    0.2723  0.4864 0.1741 0.2882   11.9    345    23    52
  0.85  0.00   40.0  4.0e-04    0.2747  0.4712 0.1814 0.2972    6.2    121     9   179
  0.85  0.00   72.0  1.2e-04    0.2469  0.4864 0.1477 0.2471    9.2    261    14    52
  0.85  0.00   72.0  4.0e-04    0.2659  0.4712 0.1677 0.2746    5.5    104     8   179
  0.93  0.00   18.0  1.2e-04    0.2918  0.5160 0.2222 0.3540   11.8    316    37    78
  0.93  0.00   18.0  4.0e-04    0.2455  0.4349 0.1811 0.2865    4.5     73    10   249
  0.93  0.00   40.0  1.2e-04    0.3005  0.5160 0.2223 0.3549   10.4    270    32    78
  0.93  0.00   40.0  4.0e-04    0.2473  0.4349 0.1802 0.2845    4.3     73     7   249
  0.93  0.00   72.0  1.2e-04    0.2712  0.5160 0.1804 0.2936    8.2    201    19    78

==========================================================================
BEST by matched_dice
==========================================================================
  threshold                0.9300
  min_confidence           0.0000
  merge_gap                40.0000
  min_area_fraction        0.0001
  matched_dice             0.3005
  matched_dice_over_truth  0.5308
  mean_paired_dice         0.6629
  foreground_dice          0.5160
  pq                       0.2223
  rq                       0.3549
  n_instances              10.4500
  spurious                 270.0000
  one_to_many              32.0000
  missed                   78.0000

Use it with:
  python scripts/predict.py --images data/test \
      --checkpoint runs/cpu_filanet_20epoch/best.pt --out submission.csv \
      --threshold 0.93 --min-confidence 0.0 \
      --merge-gap 40.0 --min-area-fraction 0.00012
```