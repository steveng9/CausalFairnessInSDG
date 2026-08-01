# Experiment results — batch `overnight-2026-07-30`

_Report generated 2026-07-30 23:55:22 UTC. Rewritten after every run, so this is current even if the sweep is still running._

## Progress

- runs recorded: **1200** (done: 1200)
- compute so far: 402 minutes

`partial` = the synthetic outcome column collapsed to a single class, so marginal-fidelity metrics were recorded but no classifier could be fitted.

## How to read this

- `fairness_gap` — worst-case |demographic parity gap| over the run's protected attributes, for a classifier trained on the synthetic data and evaluated on the **real holdout** (DECAF's DF-original reference: the case that matters in practice).
- `cond_fairness_gap` — the same, conditioned on the admissible attributes (CDP). This is the gap CF is specifically designed to remove; DP-style mechanisms target `fairness_gap`.
- `*__synth` — the same metric scored against the synthetic data's own distribution (DECAF's DF-synthetic reference, which the paper calls the uninteresting case). A big `real` vs `synth` divergence means fairness that only holds inside the synthetic data.
- `tvd_1way`/`tvd_2way` — marginal fidelity, **lower is better**. `downstream_accuracy_*` — train-on-synthetic/test-on-real, **higher is better**.
- DECAF rows have NULL epsilon: it has no DP mechanism and is the non-private reference point, not a competitor at a given epsilon.

## Headline: does each fairness mechanism actually reduce the gap?

Averaged over role configs and epsilons, within each dataset+method.

```
                                        n  fairness_gap  cond_fairness_gap  tvd_1way  downstream_accuracy_mlp  gap_sd
dataset sdg_method fairness_mechanism                                                                                
adult   decaf      cf                  15        0.0766             0.0603    0.4572                   0.6556  0.0625
                   dp                  15        0.0113             0.0107    0.4560                   0.7337  0.0239
                   ftu                 15        0.2402             0.2005    0.4566                   0.7198  0.2118
                   none                15        0.2807             0.2487    0.4564                   0.7134  0.2276
        mst        cf                  45        0.0182             0.0239    0.0106                   0.7538  0.0199
                   dp                  45        0.0024             0.0024    0.0106                   0.7469  0.0056
                   ftu                 45        0.0205             0.0185    0.0106                   0.7467  0.0221
                   none                45        0.0205             0.0185    0.0106                   0.7467  0.0221
        privbayes  cf                  45        0.1096             0.0813    0.0763                   0.7858  0.0766
                   dp                  45        0.1131             0.0837    0.0672                   0.7818  0.1011
                   ftu                 45        0.1488             0.1138    0.0756                   0.8048  0.0380
                   none                45        0.1912             0.1486    0.0729                   0.8001  0.0887
        privsyn    cf                  45        0.1330             0.1056    0.0197                   0.7791  0.1000
                   dp                  45        0.0961             0.0824    0.0241                   0.7074  0.0973
                   ftu                 45        0.1948             0.1572    0.0204                   0.7864  0.1184
                   none                45        0.2175             0.1808    0.0193                   0.7863  0.1009
compas  decaf      cf                  15        0.1553             0.1235    0.1309                   0.5936  0.1388
                   dp                  15        0.0661             0.0710    0.1313                   0.5461  0.0844
                   ftu                 15        0.1901             0.1551    0.1307                   0.6093  0.1437
                   none                15        0.2901             0.2302    0.1309                   0.6136  0.1977
        mst        cf                  45        0.2049             0.1161    0.0095                   0.6317  0.0662
                   dp                  45        0.1898             0.1055    0.0095                   0.6322  0.0595
                   ftu                 45        0.2215             0.1328    0.0096                   0.6307  0.0721
                   none                45        0.2325             0.1403    0.0096                   0.6302  0.0675
        privbayes  cf                  45        0.2117             0.1274    0.0128                   0.6256  0.0624
                   dp                  45        0.1982             0.1200    0.0130                   0.6210  0.0638
                   ftu                 45        0.2691             0.1590    0.0131                   0.6498  0.0737
                   none                45        0.3036             0.1913    0.0122                   0.6478  0.1086
        privsyn    cf                  45        0.2587             0.1856    0.0112                   0.6186  0.1236
                   dp                  45        0.3012             0.2504    0.0255                   0.5568  0.1711
                   ftu                 45        0.2728             0.1788    0.0124                   0.6362  0.1021
                   none                45        0.4566             0.3692    0.0116                   0.6309  0.1610

```

## Privacy/fairness/utility interaction (epsilon sweep)

```
                                                n  fairness_gap  cond_fairness_gap  tvd_1way  downstream_accuracy_mlp  gap_sd
dataset sdg_method epsilon fairness_mechanism                                                                                
adult   mst        1.0     cf                  15        0.0219             0.0275    0.0115                   0.7548  0.0255
                           dp                  15        0.0038             0.0036    0.0114                   0.7466  0.0075
                           ftu                 15        0.0186             0.0177    0.0115                   0.7461  0.0206
                           none                15        0.0186             0.0177    0.0115                   0.7461  0.0206
                   10.0    cf                  15        0.0177             0.0240    0.0102                   0.7541  0.0209
                           dp                  15        0.0004             0.0003    0.0102                   0.7473  0.0005
                           ftu                 15        0.0230             0.0202    0.0102                   0.7472  0.0195
                           none                15        0.0230             0.0202    0.0102                   0.7472  0.0195
                   1000.0  cf                  15        0.0150             0.0200    0.0102                   0.7525  0.0115
                           dp                  15        0.0030             0.0033    0.0102                   0.7467  0.0060
                           ftu                 15        0.0199             0.0176    0.0102                   0.7468  0.0268
                           none                15        0.0199             0.0176    0.0102                   0.7468  0.0268
        privbayes  1.0     cf                  15        0.0424             0.0316    0.1514                   0.7617  0.0433
                           dp                  15        0.0237             0.0164    0.1367                   0.7548  0.0551
                           ftu                 15        0.1369             0.1135    0.1446                   0.7895  0.0610
                           none                15        0.2502             0.2045    0.1364                   0.7758  0.1335
                   10.0    cf                  15        0.1425             0.1055    0.0647                   0.7954  0.0806
                           dp                  15        0.1662             0.1241    0.0526                   0.7956  0.0934
                           ftu                 15        0.1479             0.1074    0.0698                   0.8118  0.0114
                           none                15        0.1618             0.1209    0.0697                   0.8113  0.0311
                   1000.0  cf                  15        0.1437             0.1068    0.0128                   0.8003  0.0532
                           dp                  15        0.1495             0.1105    0.0125                   0.7951  0.0857
                           ftu                 15        0.1616             0.1204    0.0126                   0.8132  0.0187
                           none                15        0.1616             0.1204    0.0126                   0.8132  0.0187
        privsyn    1.0     cf                  15        0.0865             0.0721    0.0360                   0.7601  0.0829
                           dp                  15        0.0595             0.0552    0.0429                   0.6367  0.0922
                           ftu                 15        0.1336             0.1114    0.0366                   0.7662  0.1617
                           none                15        0.1972             0.1533    0.0333                   0.7839  0.0661
                   10.0    cf                  15        0.1087             0.0805    0.0114                   0.7767  0.0993
                           dp                  15        0.1214             0.0983    0.0180                   0.7462  0.0811
                           ftu                 15        0.2222             0.1786    0.0117                   0.7938  0.0890
                           none                15        0.1958             0.1752    0.0115                   0.7701  0.1511
                   1000.0  cf                  15        0.2039             0.1641    0.0116                   0.8005  0.0796
                           dp                  15        0.1075             0.0937    0.0114                   0.7391  0.1113
                           ftu                 15        0.2287             0.1816    0.0130                   0.7992  0.0626
                           none                15        0.2596             0.2138    0.0130                   0.8049  0.0441
compas  mst        1.0     cf                  15        0.2230             0.1312    0.0110                   0.6260  0.0733
                           dp                  15        0.1831             0.0992    0.0110                   0.6346  0.0545
                           ftu                 15        0.2448             0.1547    0.0112                   0.6231  0.0810
                           none                15        0.2647             0.1688    0.0112                   0.6201  0.0553
                   10.0    cf                  15        0.2067             0.1300    0.0089                   0.6346  0.0770
                           dp                  15        0.1936             0.1151    0.0089                   0.6286  0.0638
                           ftu                 15        0.2219             0.1455    0.0089                   0.6346  0.0814
                           none                15        0.2257             0.1438    0.0089                   0.6346  0.0792
                   1000.0  cf                  15        0.1850             0.0871    0.0087                   0.6345  0.0411
                           dp                  15        0.1927             0.1022    0.0087                   0.6334  0.0634
                           ftu                 15        0.1978             0.0981    0.0087                   0.6344  0.0444
                           none                15        0.2072             0.1083    0.0087                   0.6361  0.0562
        privbayes  1.0     cf                  15        0.1752             0.1146    0.0168                   0.5933  0.0679
                           dp                  15        0.1529             0.1037    0.0172                   0.5876  0.0700
                           ftu                 15        0.2566             0.1475    0.0176                   0.6498  0.0721
                           none                15        0.3233             0.2300    0.0154                   0.6400  0.1687
                   10.0    cf                  15        0.2287             0.1519    0.0106                   0.6303  0.0688
                           dp                  15        0.2208             0.1478    0.0106                   0.6222  0.0620
                           ftu                 15        0.2627             0.1524    0.0116                   0.6449  0.0673
                           none                15        0.2728             0.1465    0.0113                   0.6475  0.0496
                   1000.0  cf                  15        0.2312             0.1159    0.0108                   0.6532  0.0276
                           dp                  15        0.2208             0.1085    0.0112                   0.6531  0.0272
                           ftu                 15        0.2878             0.1770    0.0102                   0.6547  0.0821
                           none                15        0.3148             0.1976    0.0101                   0.6560  0.0677
        privsyn    1.0     cf                  15        0.2144             0.1392    0.0161                   0.6114  0.1121
                           dp                  15        0.2966             0.2504    0.0253                   0.5460  0.1298
                           ftu                 15        0.3285             0.2131    0.0197                   0.6432  0.1101
                           none                15        0.4643             0.3806    0.0172                   0.6337  0.1554
                   10.0    cf                  15        0.3008             0.1984    0.0088                   0.6447  0.1025
                           dp                  15        0.2505             0.2016    0.0191                   0.5545  0.1828
                           ftu                 15        0.2651             0.1457    0.0088                   0.6581  0.0732
                           none                15        0.4273             0.3194    0.0088                   0.6427  0.1333
                   1000.0  cf                  15        0.2609             0.2193    0.0087                   0.5998  0.1447
                           dp                  15        0.3564             0.2993    0.0323                   0.5699  0.1890
                           ftu                 15        0.2248             0.1775    0.0087                   0.6073  0.0969
                           none                15        0.4781             0.4076    0.0087                   0.6163  0.1953

```

## Sensitivity to the protected/admissible split

The admissible set is what CF is allowed to use to block a path, so a narrower admissible set should push CF toward DP's behaviour.

```
                                                     n  fairness_gap  cond_fairness_gap  tvd_1way  downstream_accuracy_mlp  gap_sd
dataset role_config             fairness_mechanism                                                                                
adult   prefair                 cf                  50        0.0837             0.0603    0.0794                   0.7629  0.0775
                                dp                  50        0.0817             0.0629    0.0737                   0.7320  0.1045
                                ftu                 50        0.1406             0.1028    0.0775                   0.7742  0.1183
                                none                50        0.1567             0.1178    0.0761                   0.7717  0.1315
        sex_only_broad_adm      cf                  50        0.1007             0.0785    0.0772                   0.7560  0.1016
                                dp                  50        0.0539             0.0420    0.0785                   0.7402  0.0827
                                ftu                 50        0.1398             0.1153    0.0787                   0.7728  0.1348
                                none                50        0.1505             0.1244    0.0761                   0.7717  0.1365
        sex_race_narrow_adm     cf                  50        0.0733             0.0689    0.0765                   0.7646  0.0760
                                dp                  50        0.0583             0.0499    0.0763                   0.7604  0.0845
                                ftu                 50        0.1193             0.1026    0.0769                   0.7731  0.1164
                                none                50        0.1633             0.1455    0.0772                   0.7704  0.1473
compas  prefair                 cf                  50        0.2237             0.1332    0.0231                   0.6213  0.1010
                                dp                  50        0.2295             0.1541    0.0239                   0.5873  0.1350
                                ftu                 50        0.2586             0.1380    0.0237                   0.6396  0.0921
                                none                50        0.3327             0.2107    0.0230                   0.6351  0.1606
        race_only_broad_adm     cf                  50        0.2320             0.0991    0.0232                   0.6315  0.1050
                                dp                  50        0.1706             0.0604    0.0345                   0.6180  0.0950
                                ftu                 50        0.2380             0.0955    0.0235                   0.6391  0.0990
                                none                50        0.3247             0.1834    0.0230                   0.6343  0.1630
        sex_race_age_narrow_adm cf                  50        0.1986             0.1910    0.0230                   0.6137  0.0879
                                dp                  50        0.2399             0.2351    0.0242                   0.5875  0.1396
                                ftu                 50        0.2474             0.2365    0.0236                   0.6291  0.0946
                                none                50        0.3231             0.3057    0.0232                   0.6327  0.1458

```

## Distributional Fairness axis: real vs synthetic reference

```
                                        n  max_abs_dp_gap__real  max_abs_dp_gap__synth  max_abs_cdp_gap__real  max_abs_cdp_gap__synth
dataset sdg_method fairness_mechanism                                                                                                
adult   decaf      cf                  15                0.0766                 0.1722                 0.0603                  0.0564
                   dp                  15                0.0113                 0.0014                 0.0107                  0.0014
                   ftu                 15                0.2402                 0.2259                 0.2005                  0.2172
                   none                15                0.2807                 0.2418                 0.2487                  0.2286
        mst        cf                  45                0.0182                 0.0241                 0.0239                  0.0205
                   dp                  45                0.0024                 0.0020                 0.0024                  0.0016
                   ftu                 45                0.0205                 0.0223                 0.0185                  0.0237
                   none                45                0.0205                 0.0223                 0.0185                  0.0237
        privbayes  cf                  45                0.1096                 0.0408                 0.0813                  0.0299
                   dp                  45                0.1131                 0.0189                 0.0837                  0.0183
                   ftu                 45                0.1488                 0.1294                 0.1138                  0.1104
                   none                45                0.1912                 0.1622                 0.1486                  0.1385
        privsyn    cf                  45                0.1330                 0.0878                 0.1056                  0.0655
                   dp                  45                0.0961                 0.0345                 0.0824                  0.0329
                   ftu                 45                0.1948                 0.1268                 0.1572                  0.0972
                   none                45                0.2175                 0.1510                 0.1808                  0.1253
compas  decaf      cf                  15                0.1553                 0.2007                 0.1235                  0.1191
                   dp                  15                0.0661                 0.0606                 0.0710                  0.0636
                   ftu                 15                0.1901                 0.2267                 0.1551                  0.1418
                   none                15                0.2901                 0.2766                 0.2302                  0.2258
        mst        cf                  45                0.2049                 0.1767                 0.1161                  0.0828
                   dp                  45                0.1898                 0.0655                 0.1055                  0.0691
                   ftu                 45                0.2215                 0.2350                 0.1328                  0.1391
                   none                45                0.2325                 0.2399                 0.1403                  0.1437
        privbayes  cf                  45                0.2117                 0.1364                 0.1274                  0.0953
                   dp                  45                0.1982                 0.0926                 0.1200                  0.0941
                   ftu                 45                0.2691                 0.2380                 0.1590                  0.1465
                   none                45                0.3036                 0.2936                 0.1913                  0.1905
        privsyn    cf                  45                0.2587                 0.2315                 0.1856                  0.1769
                   dp                  45                0.3012                 0.2486                 0.2504                  0.2436
                   ftu                 45                0.2728                 0.2478                 0.1788                  0.1800
                   none                45                0.4566                 0.4269                 0.3692                  0.3578

```

## Lowest fairness gap overall

Per-cell means across seeds. `n` = trials in the cell, `gap_sd` = spread of `fairness_gap` across them; a cell whose lead is smaller than its `gap_sd` is not actually ahead.

```
   dataset          role_config sdg_method fairness_mechanism   epsilon  n  fairness_gap  cond_fairness_gap  tvd_1way  downstream_accuracy_mlp  gap_sd
0    adult   sex_only_broad_adm        mst                 dp   10.0000  5        0.0001             0.0001    0.0102                   0.7473  0.0001
1    adult  sex_race_narrow_adm        mst                 dp   10.0000  5        0.0002             0.0002    0.0102                   0.7473  0.0003
2    adult   sex_only_broad_adm        mst                 dp 1000.0000  5        0.0004             0.0007    0.0102                   0.7467  0.0007
3    adult  sex_race_narrow_adm        mst                 dp 1000.0000  5        0.0005             0.0007    0.0102                   0.7467  0.0006
4    adult              prefair        mst                 dp   10.0000  5        0.0007             0.0005    0.0102                   0.7473  0.0007
5    adult   sex_only_broad_adm        mst                 dp    1.0000  5        0.0007             0.0004    0.0114                   0.7466  0.0010
6    adult  sex_race_narrow_adm        mst                 dp    1.0000  5        0.0010             0.0009    0.0114                   0.7466  0.0011
7    adult   sex_only_broad_adm        mst                 cf 1000.0000  5        0.0023             0.0017    0.0102                   0.7486  0.0023
8    adult   sex_only_broad_adm    privsyn                 dp    1.0000  5        0.0039             0.0023    0.0398                   0.5635  0.0064
9    adult   sex_only_broad_adm        mst                 cf   10.0000  5        0.0047             0.0025    0.0102                   0.7487  0.0049
10   adult   sex_only_broad_adm      decaf                 dp       NaN  5        0.0061             0.0054    0.4560                   0.7328  0.0090
11   adult   sex_only_broad_adm        mst                 cf    1.0000  5        0.0064             0.0036    0.0115                   0.7514  0.0095
12   adult              prefair        mst                 dp 1000.0000  5        0.0079             0.0085    0.0102                   0.7467  0.0088
13   adult              prefair        mst                 dp    1.0000  5        0.0098             0.0096    0.0114                   0.7466  0.0114
14   adult   sex_only_broad_adm        mst                ftu    1.0000  5        0.0099             0.0084    0.0115                   0.7461  0.0134
15   adult   sex_only_broad_adm        mst               none    1.0000  5        0.0099             0.0084    0.0115                   0.7461  0.0134
16   adult   sex_only_broad_adm  privbayes                 dp    1.0000  5        0.0113             0.0075    0.1467                   0.7514  0.0252
17   adult  sex_race_narrow_adm        mst                ftu    1.0000  5        0.0130             0.0139    0.0115                   0.7461  0.0133
18   adult  sex_race_narrow_adm        mst               none    1.0000  5        0.0130             0.0139    0.0115                   0.7461  0.0133
19   adult  sex_race_narrow_adm      decaf                 dp       NaN  5        0.0133             0.0131    0.4559                   0.7341  0.0291

```

## Lowest fairness gap among configs with usable accuracy (>= median, 0.661)

Guards against the degenerate 'perfectly fair because the model predicts one class' solution.

```
   dataset          role_config sdg_method fairness_mechanism   epsilon  n  fairness_gap  cond_fairness_gap  tvd_1way  downstream_accuracy_mlp  gap_sd
0    adult   sex_only_broad_adm        mst                 dp   10.0000  5        0.0001             0.0001    0.0102                   0.7473  0.0001
1    adult  sex_race_narrow_adm        mst                 dp   10.0000  5        0.0002             0.0002    0.0102                   0.7473  0.0003
2    adult   sex_only_broad_adm        mst                 dp 1000.0000  5        0.0004             0.0007    0.0102                   0.7467  0.0007
3    adult  sex_race_narrow_adm        mst                 dp 1000.0000  5        0.0005             0.0007    0.0102                   0.7467  0.0006
4    adult              prefair        mst                 dp   10.0000  5        0.0007             0.0005    0.0102                   0.7473  0.0007
5    adult   sex_only_broad_adm        mst                 dp    1.0000  5        0.0007             0.0004    0.0114                   0.7466  0.0010
6    adult  sex_race_narrow_adm        mst                 dp    1.0000  5        0.0010             0.0009    0.0114                   0.7466  0.0011
7    adult   sex_only_broad_adm        mst                 cf 1000.0000  5        0.0023             0.0017    0.0102                   0.7486  0.0023
8    adult   sex_only_broad_adm        mst                 cf   10.0000  5        0.0047             0.0025    0.0102                   0.7487  0.0049
9    adult   sex_only_broad_adm      decaf                 dp       NaN  5        0.0061             0.0054    0.4560                   0.7328  0.0090
10   adult   sex_only_broad_adm        mst                 cf    1.0000  5        0.0064             0.0036    0.0115                   0.7514  0.0095
11   adult              prefair        mst                 dp 1000.0000  5        0.0079             0.0085    0.0102                   0.7467  0.0088
12   adult              prefair        mst                 dp    1.0000  5        0.0098             0.0096    0.0114                   0.7466  0.0114
13   adult   sex_only_broad_adm        mst               none    1.0000  5        0.0099             0.0084    0.0115                   0.7461  0.0134
14   adult   sex_only_broad_adm        mst                ftu    1.0000  5        0.0099             0.0084    0.0115                   0.7461  0.0134
15   adult   sex_only_broad_adm  privbayes                 dp    1.0000  5        0.0113             0.0075    0.1467                   0.7514  0.0252
16   adult  sex_race_narrow_adm        mst               none    1.0000  5        0.0130             0.0139    0.0115                   0.7461  0.0133
17   adult  sex_race_narrow_adm        mst                ftu    1.0000  5        0.0130             0.0139    0.0115                   0.7461  0.0133
18   adult  sex_race_narrow_adm      decaf                 dp       NaN  5        0.0133             0.0131    0.4559                   0.7341  0.0291
19   adult              prefair      decaf                 dp       NaN  5        0.0145             0.0135    0.4559                   0.7341  0.0319

```
