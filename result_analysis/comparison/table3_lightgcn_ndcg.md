# Table 3 Extended: LightGCN Backbone NDCG@K Results

This table shows NDCG@K results for LightGCN backbone on Health and Electronic datasets, extending the analysis beyond MF backbone to test generalizability.

| Dataset | Method | NDCG@5 | Imp% | NDCG@10 | Imp% | NDCG@20 | Imp% | NDCG@50 | Imp% | NDCG@75 | Imp% | NDCG@100 | Imp% |
|---------|--------|----------|------|----------|------|----------|------|----------|------|----------|------|----------|------|
| Electronic | AdvInfoNCE | 0.0350 |  | 0.0427 |  | 0.0531 |  | 0.0697 |  | 0.0786 |  | 0.0851 |  |
| Electronic | BPR | 0.0124 |  | 0.0165 |  | 0.0224 |  | 0.0338 |  | 0.0402 |  | 0.0452 |  |
| Electronic | BSL | 0.0345 |  | 0.0425 |  | 0.0525 |  | 0.0697 |  | 0.0782 |  | 0.0849 |  |
| Electronic | GuidedRec | 0.0211 |  | 0.0277 |  | 0.0364 |  | 0.0524 |  | 0.0605 |  | 0.0673 |  |
| Electronic | LLPAUC | 0.0350 |  | 0.0429 |  | 0.0531 |  | 0.0698 |  | 0.0781 |  | 0.0845 |  |
| Electronic | PSL | 0.0356 |  | 0.0433 |  | 0.0530 |  | 0.0695 |  | 0.0782 |  | 0.0846 |  |
| Electronic | SLatK | 0.0402 | <span style="color:blue">+12.74% (vs PSL)</span> | 0.0483 | <span style="color:blue">+11.52% (vs PSL)</span> | 0.0593 | <span style="color:blue">+11.58% (vs LLPAUC)</span> | 0.0762 | <span style="color:blue">+9.08% (vs LLPAUC)</span> | 0.0846 | <span style="color:blue">+7.70% (vs AdvInfoNCE)</span> | 0.0908 | <span style="color:blue">+6.80% (vs AdvInfoNCE)</span> |
| Electronic | SONGatK | 0.0301 |  | 0.0373 |  | 0.0472 |  | 0.0630 |  | 0.0715 |  | 0.0778 |  |
| Electronic | Softmax | 0.0344 |  | 0.0425 |  | 0.0524 |  | 0.0696 |  | 0.0782 |  | 0.0848 |  |
|---------|--------|----------|------|----------|------|----------|------|----------|------|----------|------|----------|------|
| Health | AdvInfoNCE | 0.0918 |  | 0.1035 |  | 0.1254 |  | 0.1610 |  | 0.1778 |  | 0.1917 |  |
| Health | BPR | 0.0465 |  | 0.0559 |  | 0.0744 |  | 0.1068 |  | 0.1242 |  | 0.1370 |  |
| Health | BSL | 0.0915 |  | 0.1022 |  | 0.1240 |  | 0.1604 |  | 0.1780 |  | 0.1915 |  |
| Health | GuidedRec | 0.0668 |  | 0.0785 |  | 0.1011 |  | 0.1408 |  | 0.1594 |  | 0.1719 |  |
| Health | LLPAUC | 0.0885 |  | 0.1034 |  | 0.1271 |  | 0.1653 |  | 0.1850 |  | 0.1970 |  |
| Health | PSL | 0.0939 |  | 0.1043 |  | 0.1252 |  | 0.1616 |  | 0.1785 |  | 0.1911 |  |
| Health | SLatK | 0.1058 | <span style="color:blue">+12.64% (vs PSL)</span> | 0.1168 | <span style="color:blue">+11.92% (vs PSL)</span> | 0.1384 | <span style="color:blue">+8.86% (vs LLPAUC)</span> | 0.1729 | <span style="color:blue">+4.61% (vs LLPAUC)</span> | 0.1905 | <span style="color:blue">+2.93% (vs LLPAUC)</span> | 0.2025 | <span style="color:blue">+2.81% (vs LLPAUC)</span> |
| Health | SONGatK | 0.0836 |  | 0.0940 |  | 0.1157 |  | 0.1506 |  | 0.1676 |  | 0.1800 |  |
| Health | Softmax | 0.0914 |  | 0.1022 |  | 0.1240 |  | 0.1603 |  | 0.1780 |  | 0.1915 |  |

## Notes
- NDCG@K: Our experiment results using LightGCN backbone
- Imp%: Our improvement percentage of SL@K relative to best baseline for LightGCN backbone (only shown for SLatK method)
- This table extends Table 3 analysis to LightGCN backbone to test the generalizability of the SL@K method
- Only Health and Electronic datasets are included (same as original Table 3)
- Results help understand whether SL@K improvements are specific to MF backbone or generalizable to other architectures
