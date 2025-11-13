# Experiment Results Comparison

The following is a comparison between our experiment results and the author's results:

| dataset | backbone | loss | Our_Recall@20 | Author_Recall@20 | Recall_Diff | Our_NDCG@20 | Author_NDCG@20 | NDCG_Diff | Author_Recall_Imp% | Author_NDCG_Imp% | Our_Recall_Imp% | Our_NDCG_Imp% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| book | MF | AdvInfoNCE | 0.0908 | 0.1557 | <span style="color:red">-0.0649</span> | 0.0679 | 0.1172 | <span style="color:red">-0.0493</span> |  |  |  |  |
| book | MF | BPR | 0.0235 | 0.0665 | <span style="color:red">-0.0430</span> | 0.0171 | 0.0453 | <span style="color:red">-0.0282</span> |  |  |  |  |
| book | MF | BSL | 0.0916 | 0.1563 | <span style="color:red">-0.0647</span> | 0.0687 | 0.1212 | <span style="color:red">-0.0525</span> |  |  |  |  |
| book | MF | GuidedRec | 0.0388 | 0.0518 | <span style="color:red">-0.0130</span> | 0.0273 | 0.0361 | -0.0088 |  |  |  |  |
| book | MF | LLPAUC | 0.0851 | 0.1150 | <span style="color:red">-0.0299</span> | 0.0610 | 0.0811 | <span style="color:red">-0.0201</span> |  |  |  |  |
| book | MF | PSL | 0.0902 | 0.1569 | <span style="color:red">-0.0667</span> | 0.0676 | 0.1227 | <span style="color:red">-0.0551</span> |  |  |  |  |
| book | MF | SLatK | 0.0923 | 0.1612 | <span style="color:red">-0.0689</span> | 0.0694 | 0.1269 | <span style="color:red">-0.0575</span> | <span style="color:blue">+2.74% (vs PSL)</span> | <span style="color:blue">+3.42% (vs PSL)</span> | <span style="color:blue">+0.76% (vs BSL)</span> | <span style="color:blue">+0.96% (vs BSL)</span> |
| book | MF | SONGatK | 0.0847 | 0.0747 | 0.0100 | 0.0620 | 0.0542 | 0.0078 |  |  |  |  |
| book | MF | Softmax | 0.0913 | 0.1559 | <span style="color:red">-0.0646</span> | 0.0683 | 0.1210 | <span style="color:red">-0.0527</span> |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| book | XSimGCL | BPR | 0.0678 | 0.1269 | <span style="color:red">-0.0591</span> | 0.0491 | 0.0905 | <span style="color:red">-0.0414</span> |  |  |  |  |
| book | XSimGCL | LLPAUC | 0.0765 | 0.1363 | <span style="color:red">-0.0598</span> | 0.0544 | 0.1008 | <span style="color:red">-0.0464</span> |  |  |  |  |
| book | XSimGCL | PSL | 0.0800 | 0.1571 | <span style="color:red">-0.0771</span> | 0.0590 | 0.1228 | <span style="color:red">-0.0638</span> |  |  |  |  |
| book | XSimGCL | SLatK | 0.0752 | 0.1624 | <span style="color:red">-0.0872</span> | 0.0546 | 0.1277 | <span style="color:red">-0.0731</span> | <span style="color:blue">+3.37% (vs PSL)</span> | <span style="color:blue">+3.99% (vs PSL)</span> | <span style="color:blue">-7.68% (vs Softmax)</span> | <span style="color:blue">-8.47% (vs Softmax)</span> |
| book | XSimGCL | Softmax | 0.0815 | 0.1549 | <span style="color:red">-0.0734</span> | 0.0597 | 0.1207 | <span style="color:red">-0.0610</span> |  |  |  |  |
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
| gowalla | LightGCN | AdvInfoNCE | 0.1454 | 0.2066 | <span style="color:red">-0.0612</span> | 0.1116 | 0.1625 | <span style="color:red">-0.0509</span> |  |  |  |  |
| gowalla | LightGCN | BPR | 0.0641 | 0.1745 | <span style="color:red">-0.1104</span> | 0.0437 | 0.1402 | <span style="color:red">-0.0965</span> |  |  |  |  |
| gowalla | LightGCN | BSL | 0.1460 | 0.2069 | <span style="color:red">-0.0609</span> | 0.1130 | 0.1628 | <span style="color:red">-0.0498</span> |  |  |  |  |
| gowalla | LightGCN | GuidedRec | 0.0864 | 0.0921 | -0.0057 | 0.0643 | 0.0686 | -0.0043 |  |  |  |  |
| gowalla | LightGCN | LLPAUC | 0.1381 | 0.1616 | <span style="color:red">-0.0235</span> | 0.1023 | 0.1192 | <span style="color:red">-0.0169</span> |  |  |  |  |
| gowalla | LightGCN | PSL | 0.1419 | 0.2086 | <span style="color:red">-0.0667</span> | 0.1097 | 0.1648 | <span style="color:red">-0.0551</span> |  |  |  |  |
| gowalla | LightGCN | SLatK | 0.1514 | 0.2128 | <span style="color:red">-0.0614</span> | 0.1174 | 0.1729 | <span style="color:red">-0.0555</span> | <span style="color:blue">+2.01% (vs PSL)</span> | <span style="color:blue">+4.92% (vs PSL)</span> | <span style="color:blue">+1.63% (vs SONGatK)</span> | <span style="color:blue">+1.52% (vs SONGatK)</span> |
| gowalla | LightGCN | SONGatK | 0.1489 | 0.1261 | <span style="color:green">0.0228</span> | 0.1156 | 0.0968 | <span style="color:green">0.0188</span> |  |  |  |  |
| gowalla | LightGCN | Softmax | 0.1461 | 0.2068 | <span style="color:red">-0.0607</span> | 0.1129 | 0.1628 | <span style="color:red">-0.0499</span> |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gowalla | MF | AdvInfoNCE | 0.1469 | 0.2067 | <span style="color:red">-0.0598</span> | 0.1127 | 0.1627 | <span style="color:red">-0.0500</span> |  |  |  |  |
| gowalla | MF | BPR | 0.0642 | 0.1355 | <span style="color:red">-0.0713</span> | 0.0458 | 0.1111 | <span style="color:red">-0.0653</span> |  |  |  |  |
| gowalla | MF | BSL | 0.1461 | 0.2071 | <span style="color:red">-0.0610</span> | 0.1131 | 0.1630 | <span style="color:red">-0.0499</span> |  |  |  |  |
| gowalla | MF | GuidedRec | 0.0997 | 0.1135 | <span style="color:red">-0.0138</span> | 0.0750 | 0.0863 | <span style="color:red">-0.0113</span> |  |  |  |  |
| gowalla | MF | LLPAUC | 0.1345 | 0.1610 | <span style="color:red">-0.0265</span> | 0.1007 | 0.1189 | <span style="color:red">-0.0182</span> |  |  |  |  |
| gowalla | MF | PSL | 0.1434 | 0.2089 | <span style="color:red">-0.0655</span> | 0.1105 | 0.1647 | <span style="color:red">-0.0542</span> |  |  |  |  |
| gowalla | MF | SLatK | 0.1504 | 0.2121 | <span style="color:red">-0.0617</span> | 0.1177 | 0.1709 | <span style="color:red">-0.0532</span> | <span style="color:blue">+1.53% (vs PSL)</span> | <span style="color:blue">+3.76% (vs PSL)</span> | <span style="color:blue">+2.43% (vs AdvInfoNCE)</span> | <span style="color:blue">+3.27% (vs SONGatK)</span> |
| gowalla | MF | SONGatK | 0.1464 | 0.1237 | <span style="color:green">0.0227</span> | 0.1140 | 0.0970 | <span style="color:green">0.0170</span> |  |  |  |  |
| gowalla | MF | Softmax | 0.1461 | 0.2064 | <span style="color:red">-0.0603</span> | 0.1131 | 0.1624 | <span style="color:red">-0.0493</span> |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gowalla | XSimGCL | AdvInfoNCE | 0.1262 | 0.2010 | <span style="color:red">-0.0748</span> | 0.1017 | 0.1564 | <span style="color:red">-0.0547</span> |  |  |  |  |
| gowalla | XSimGCL | BPR | 0.1029 | 0.1966 | <span style="color:red">-0.0937</span> | 0.0747 | 0.1570 | <span style="color:red">-0.0823</span> |  |  |  |  |
| gowalla | XSimGCL | BSL | 0.1267 | 0.2037 | <span style="color:red">-0.0770</span> | 0.0957 | 0.1597 | <span style="color:red">-0.0640</span> |  |  |  |  |
| gowalla | XSimGCL | GuidedRec | 0.1124 | 0.1685 | <span style="color:red">-0.0561</span> | 0.0816 | 0.1277 | <span style="color:red">-0.0461</span> |  |  |  |  |
| gowalla | XSimGCL | LLPAUC | 0.1181 | 0.1632 | <span style="color:red">-0.0451</span> | 0.0856 | 0.1200 | <span style="color:red">-0.0344</span> |  |  |  |  |
| gowalla | XSimGCL | PSL | 0.1240 | 0.2037 | <span style="color:red">-0.0797</span> | 0.0943 | 0.1593 | <span style="color:red">-0.0650</span> |  |  |  |  |
| gowalla | XSimGCL | SLatK | 0.1242 | 0.2095 | <span style="color:red">-0.0853</span> | 0.0934 | 0.1717 | <span style="color:red">-0.0783</span> | <span style="color:blue">+2.85% (vs BSL)</span> | <span style="color:blue">+7.51% (vs BSL)</span> | <span style="color:blue">-2.07% (vs Softmax)</span> | <span style="color:blue">-8.07% (vs AdvInfoNCE)</span> |
| gowalla | XSimGCL | SONGatK | 0.0956 | 0.1367 | <span style="color:red">-0.0411</span> | 0.0689 | 0.0985 | <span style="color:red">-0.0296</span> |  |  |  |  |
| gowalla | XSimGCL | Softmax | 0.1268 | 0.2005 | <span style="color:red">-0.0737</span> | 0.0960 | 0.1570 | <span style="color:red">-0.0610</span> |  |  |  |  |
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
