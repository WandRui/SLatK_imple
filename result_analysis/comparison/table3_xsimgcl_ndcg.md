# Table 3 Extended: XSimGCL Backbone NDCG@K Results

This table shows NDCG@K results for XSimGCL backbone on Health and Electronic datasets, extending the analysis beyond MF backbone to test generalizability.

| Dataset | Method | NDCG@5 | Imp% | NDCG@10 | Imp% | NDCG@20 | Imp% | NDCG@50 | Imp% | NDCG@75 | Imp% | NDCG@100 | Imp% |
|---------|--------|----------|------|----------|------|----------|------|----------|------|----------|------|----------|------|
| Electronic | AdvInfoNCE | 0.0308 |  | 0.0383 |  | 0.0479 |  | 0.0637 |  | 0.0714 |  | 0.0780 |  |
| Electronic | BPR | 0.0169 |  | 0.0214 |  | 0.0274 |  | 0.0385 |  | 0.0444 |  | 0.0490 |  |
| Electronic | BSL | 0.0327 |  | 0.0402 |  | 0.0496 |  | 0.0659 |  | 0.0741 |  | 0.0806 |  |
| Electronic | GuidedRec | 0.0279 |  | 0.0350 |  | 0.0443 |  | 0.0607 |  | 0.0693 |  | 0.0757 |  |
| Electronic | LLPAUC | 0.0271 |  | 0.0339 |  | 0.0437 |  | 0.0600 |  | 0.0686 |  | 0.0754 |  |
| Electronic | PSL | 0.0323 |  | 0.0392 |  | 0.0489 |  | 0.0647 |  | 0.0728 |  | 0.0790 |  |
| Electronic | SLatK | 0.0351 | <span style="color:blue">+7.48% (vs BSL)</span> | 0.0419 | <span style="color:blue">+4.16% (vs BSL)</span> | 0.0503 | <span style="color:blue">+1.31% (vs BSL)</span> | 0.0641 | <span style="color:blue">-2.75% (vs BSL)</span> | 0.0711 | <span style="color:blue">-4.00% (vs BSL)</span> | 0.0765 | <span style="color:blue">-5.18% (vs BSL)</span> |
| Electronic | SONGatK | 0.0175 |  | 0.0223 |  | 0.0292 |  | 0.0404 |  | 0.0471 |  | 0.0521 |  |
| Electronic | Softmax | 0.0308 |  | 0.0383 |  | 0.0477 |  | 0.0638 |  | 0.0719 |  | 0.0781 |  |
|---------|--------|----------|------|----------|------|----------|------|----------|------|----------|------|----------|------|
| Health | AdvInfoNCE | 0.0833 |  | 0.0949 |  | 0.1162 |  | 0.1512 |  | 0.1694 |  | 0.1819 |  |
| Health | BPR | 0.0589 |  | 0.0671 |  | 0.0843 |  | 0.1167 |  | 0.1344 |  | 0.1465 |  |
| Health | BSL | 0.0872 |  | 0.0991 |  | 0.1223 |  | 0.1563 |  | 0.1732 |  | 0.1853 |  |
| Health | GuidedRec | 0.0772 |  | 0.0879 |  | 0.1103 |  | 0.1484 |  | 0.1663 |  | 0.1788 |  |
| Health | LLPAUC | 0.0820 |  | 0.0944 |  | 0.1170 |  | 0.1552 |  | 0.1734 |  | 0.1874 |  |
| Health | PSL | 0.0863 |  | 0.0977 |  | 0.1192 |  | 0.1544 |  | 0.1704 |  | 0.1834 |  |
| Health | SLatK | 0.0914 | <span style="color:blue">+4.81% (vs BSL)</span> | 0.1025 | <span style="color:blue">+3.45% (vs BSL)</span> | 0.1222 | <span style="color:blue">-0.10% (vs BSL)</span> | 0.1573 | <span style="color:blue">+0.64% (vs BSL)</span> | 0.1744 | <span style="color:blue">+0.61% (vs LLPAUC)</span> | 0.1864 | <span style="color:blue">-0.52% (vs LLPAUC)</span> |
| Health | SONGatK | 0.0675 |  | 0.0757 |  | 0.0929 |  | 0.1266 |  | 0.1442 |  | 0.1557 |  |
| Health | Softmax | 0.0845 |  | 0.0965 |  | 0.1183 |  | 0.1538 |  | 0.1710 |  | 0.1835 |  |

## Notes
- NDCG@K: Our experiment results using XSimGCL backbone
- Imp%: Our improvement percentage of SL@K relative to best baseline for XSimGCL backbone (only shown for SLatK method)
- This table extends Table 3 analysis to XSimGCL backbone to test the generalizability of the SL@K method
- Only Health and Electronic datasets are included (same as original Table 3)
- Results help understand whether SL@K improvements are specific to MF backbone or generalizable to other architectures
