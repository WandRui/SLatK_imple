# Experiment Results Comparison

The following is a comparison between our experiment results and the author's results:

| dataset | backbone | loss | Our_Recall@20 | Author_Recall@20 | Recall_Diff | Our_NDCG@20 | Author_NDCG@20 | NDCG_Diff | Author_Recall_Imp% | Author_NDCG_Imp% | Our_Recall_Imp% | Our_NDCG_Imp% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electronic | LightGCN | AdvInfoNCE | 0.0836 | 0.0823 | 0.0013 | 0.0531 | 0.0528 | 0.0003 |  |  |  |  |
| electronic | LightGCN | BPR | 0.0390 | 0.0813 | <span style="color:red">-0.0423</span> | 0.0224 | 0.0524 | <span style="color:red">-0.0300</span> |  |  |  |  |
| electronic | LightGCN | BSL | 0.0823 | 0.0823 | 0.0000 | 0.0525 | 0.0526 | -0.0001 |  |  |  |  |
| electronic | LightGCN | GuidedRec | 0.0614 | 0.0657 | -0.0043 | 0.0364 | 0.0393 | -0.0029 |  |  |  |  |
| electronic | LightGCN | LLPAUC | 0.0830 | 0.0831 | -0.0001 | 0.0531 | 0.0507 | 0.0024 |  |  |  |  |
| electronic | LightGCN | PSL | 0.0822 | 0.0830 | -0.0008 | 0.0530 | 0.0536 | -0.0006 |  |  |  |  |
| electronic | LightGCN | SLatK | 0.0897 | 0.0903 | -0.0006 | 0.0589 | 0.0591 | -0.0002 | <span style="color:blue">+8.66% (vs LLPAUC)</span> | <span style="color:blue">+10.26% (vs PSL)</span> | <span style="color:blue">+7.25% (vs AdvInfoNCE)</span> | <span style="color:blue">+10.92% (vs LLPAUC)</span> |
| electronic | LightGCN | SONGatK | 0.0755 | 0.0816 | -0.0061 | 0.0472 | 0.0511 | -0.0039 |  |  |  |  |
| electronic | LightGCN | Softmax | 0.0823 | 0.0823 | 0.0000 | 0.0524 | 0.0526 | -0.0002 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electronic | MF | AdvInfoNCE | 0.0814 | 0.0829 | -0.0015 | 0.0521 | 0.0527 | -0.0006 |  |  |  |  |
| electronic | MF | BPR | 0.0379 | 0.0816 | <span style="color:red">-0.0437</span> | 0.0218 | 0.0527 | <span style="color:red">-0.0309</span> |  |  |  |  |
| electronic | MF | BSL | 0.0832 | 0.0834 | -0.0002 | 0.0528 | 0.0530 | -0.0002 |  |  |  |  |
| electronic | MF | GuidedRec | 0.0607 | 0.0644 | -0.0037 | 0.0359 | 0.0385 | -0.0026 |  |  |  |  |
| electronic | MF | LLPAUC | 0.0817 | 0.0821 | -0.0004 | 0.0498 | 0.0499 | -0.0001 |  |  |  |  |
| electronic | MF | PSL | 0.0835 | 0.0838 | -0.0003 | 0.0531 | 0.0541 | -0.0010 |  |  |  |  |
| electronic | MF | SLatK | 0.0896 | 0.0901 | -0.0005 | 0.0583 | 0.0590 | -0.0007 | <span style="color:blue">+7.52% (vs PSL)</span> | <span style="color:blue">+9.06% (vs PSL)</span> | <span style="color:blue">+7.34% (vs PSL)</span> | <span style="color:blue">+9.78% (vs PSL)</span> |
| electronic | MF | SONGatK | 0.0741 | 0.0708 | 0.0033 | 0.0459 | 0.0444 | 0.0015 |  |  |  |  |
| electronic | MF | Softmax | 0.0825 | 0.0821 | 0.0004 | 0.0525 | 0.0529 | -0.0004 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| electronic | XSimGCL | AdvInfoNCE | 0.0761 | 0.0776 | -0.0015 | 0.0479 | 0.0489 | -0.0010 |  |  |  |  |
| electronic | XSimGCL | BPR | 0.0450 | 0.0777 | <span style="color:red">-0.0327</span> | 0.0274 | 0.0508 | <span style="color:red">-0.0234</span> |  |  |  |  |
| electronic | XSimGCL | BSL | 0.0778 | 0.0800 | -0.0022 | 0.0496 | 0.0507 | -0.0011 |  |  |  |  |
| electronic | XSimGCL | GuidedRec | 0.0713 | 0.0760 | -0.0047 | 0.0443 | 0.0473 | -0.0030 |  |  |  |  |
| electronic | XSimGCL | LLPAUC | 0.0711 | 0.0781 | -0.0070 | 0.0437 | 0.0481 | -0.0044 |  |  |  |  |
| electronic | XSimGCL | PSL | 0.0768 | 0.0801 | -0.0033 | 0.0489 | 0.0507 | -0.0018 |  |  |  |  |
| electronic | XSimGCL | SLatK | 0.0753 | 0.0869 | <span style="color:red">-0.0116</span> | 0.0496 | 0.0571 | -0.0075 | <span style="color:blue">+8.49% (vs PSL)</span> | <span style="color:blue">+12.40% (vs BPR)</span> | <span style="color:blue">-3.19% (vs BSL)</span> | <span style="color:blue">-0.02% (vs BSL)</span> |
| electronic | XSimGCL | SONGatK | 0.0486 | 0.0525 | -0.0039 | 0.0292 | 0.0320 | -0.0028 |  |  |  |  |
| electronic | XSimGCL | Softmax | 0.0752 | 0.0772 | -0.0020 | 0.0477 | 0.0490 | -0.0013 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| health | LightGCN | AdvInfoNCE | 0.1715 | 0.1706 | 0.0009 | 0.1254 | 0.1264 | -0.0010 |  |  |  |  |
| health | LightGCN | BPR | 0.1122 | 0.1618 | <span style="color:red">-0.0496</span> | 0.0744 | 0.1203 | <span style="color:red">-0.0459</span> |  |  |  |  |
| health | LightGCN | BSL | 0.1687 | 0.1691 | -0.0004 | 0.1240 | 0.1236 | 0.0004 |  |  |  |  |
| health | LightGCN | GuidedRec | 0.1500 | 0.1550 | -0.0050 | 0.1011 | 0.1073 | -0.0062 |  |  |  |  |
| health | LightGCN | LLPAUC | 0.1814 | 0.1685 | <span style="color:green">0.0129</span> | 0.1271 | 0.1207 | 0.0064 |  |  |  |  |
| health | LightGCN | PSL | 0.1684 | 0.1701 | -0.0017 | 0.1252 | 0.1270 | -0.0018 |  |  |  |  |
| health | LightGCN | SLatK | 0.1793 | 0.1783 | 0.0010 | 0.1369 | 0.1371 | -0.0002 | <span style="color:blue">+4.51% (vs AdvInfoNCE)</span> | <span style="color:blue">+7.95% (vs PSL)</span> | <span style="color:blue">-1.18% (vs LLPAUC)</span> | <span style="color:blue">+7.66% (vs LLPAUC)</span> |
| health | LightGCN | SONGatK | 0.1614 | 0.1353 | <span style="color:green">0.0261</span> | 0.1157 | 0.0960 | <span style="color:green">0.0197</span> |  |  |  |  |
| health | LightGCN | Softmax | 0.1687 | 0.1691 | -0.0004 | 0.1240 | 0.1235 | 0.0005 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| health | MF | AdvInfoNCE | 0.1664 | 0.1659 | 0.0005 | 0.1238 | 0.1237 | 0.0001 |  |  |  |  |
| health | MF | BPR | 0.1147 | 0.1627 | <span style="color:red">-0.0480</span> | 0.0759 | 0.1234 | <span style="color:red">-0.0475</span> |  |  |  |  |
| health | MF | BSL | 0.1675 | 0.1719 | -0.0044 | 0.1242 | 0.1261 | -0.0019 |  |  |  |  |
| health | MF | GuidedRec | 0.1465 | 0.1568 | <span style="color:red">-0.0103</span> | 0.1016 | 0.1093 | -0.0077 |  |  |  |  |
| health | MF | LLPAUC | 0.1684 | 0.1644 | 0.0040 | 0.1228 | 0.1209 | 0.0019 |  |  |  |  |
| health | MF | PSL | 0.1696 | 0.1718 | -0.0022 | 0.1252 | 0.1268 | -0.0016 |  |  |  |  |
| health | MF | SLatK | 0.1804 | 0.1823 | -0.0019 | 0.1371 | 0.1390 | -0.0019 | <span style="color:blue">+6.05% (vs Softmax)</span> | <span style="color:blue">+9.62% (vs PSL)</span> | <span style="color:blue">+6.33% (vs PSL)</span> | <span style="color:blue">+9.54% (vs PSL)</span> |
| health | MF | SONGatK | 0.1599 | 0.0874 | <span style="color:green">0.0725</span> | 0.1153 | 0.0650 | <span style="color:green">0.0503</span> |  |  |  |  |
| health | MF | Softmax | 0.1675 | 0.1719 | -0.0044 | 0.1242 | 0.1261 | -0.0019 |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| health | XSimGCL | AdvInfoNCE | 0.1619 | 0.1499 | <span style="color:green">0.0120</span> | 0.1162 | 0.1072 | 0.0090 |  |  |  |  |
| health | XSimGCL | BPR | 0.1194 | 0.1496 | <span style="color:red">-0.0302</span> | 0.0843 | 0.1108 | <span style="color:red">-0.0265</span> |  |  |  |  |
| health | XSimGCL | BSL | 0.1698 | 0.1649 | 0.0049 | 0.1223 | 0.1201 | 0.0022 |  |  |  |  |
| health | XSimGCL | GuidedRec | 0.1574 | 0.1539 | 0.0035 | 0.1103 | 0.1088 | 0.0015 |  |  |  |  |
| health | XSimGCL | LLPAUC | 0.1651 | 0.1519 | <span style="color:green">0.0132</span> | 0.1170 | 0.1083 | 0.0087 |  |  |  |  |
| health | XSimGCL | PSL | 0.1642 | 0.1579 | 0.0063 | 0.1192 | 0.1143 | 0.0049 |  |  |  |  |
| health | XSimGCL | SLatK | 0.1636 | 0.1753 | <span style="color:red">-0.0117</span> | 0.1215 | 0.1332 | <span style="color:red">-0.0117</span> | <span style="color:blue">+6.31% (vs BSL)</span> | <span style="color:blue">+10.91% (vs BSL)</span> | <span style="color:blue">-3.61% (vs BSL)</span> | <span style="color:blue">-0.63% (vs BSL)</span> |
| health | XSimGCL | SONGatK | 0.1281 | 0.1378 | -0.0097 | 0.0929 | 0.0948 | -0.0019 |  |  |  |  |
| health | XSimGCL | Softmax | 0.1630 | 0.1534 | 0.0096 | 0.1183 | 0.1113 | 0.0070 |  |  |  |  |

## Notes
- Our_Recall@20/Our_NDCG@20: Our experiment results
- Author_Recall@20/Author_NDCG@20: Author's experiment results
- Recall_Diff/NDCG_Diff: Difference (our results - author's results)
- Author_Recall_Imp%/Author_NDCG_Imp%: Author's improvement percentage of SL@K relative to best baseline (only shown for SLatK method)
- Our_Recall_Imp%/Our_NDCG_Imp%: Our reproduction improvement percentage of SL@K relative to best baseline (only shown for SLatK method)
- Positive values indicate our results are better, negative values indicate author's results are better
- Results with absolute difference ≤ 0.01 are not colored, differences > 0.01 are marked green, differences < -0.01 are marked red
