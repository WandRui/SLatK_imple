# Table 4 Extended: LightGCN Backbone SL@K Performance Analysis

This table shows the performance of SL@K method with different K values using LightGCN backbone on Health and Electronic datasets, extending the analysis beyond MF backbone.

| Dataset | Metric | SL@5 | SL@10 | SL@20 | SL@50 | SL@75 | SL@100 |
|---------|--------|----------|----------|----------|----------|----------|----------|
| Electronic | NDCG@5 | 0.0005 | 0.0006 | 0.0005 | 0.0392 | 0.0398 | 0.0005 |
| Electronic | NDCG@10 | 0.0007 | 0.0007 | 0.0008 | 0.0472 | 0.0478 | 0.0008 |
| Electronic | NDCG@20 | 0.0010 | 0.0011 | 0.0011 | 0.0582 | 0.0589 | 0.0011 |
| Electronic | NDCG@50 | 0.0018 | 0.0019 | 0.0020 | 0.0752 | 0.0759 | 0.0020 |
| Electronic | NDCG@75 | 0.0023 | 0.0024 | 0.0029 | 0.0839 | 0.0845 | 0.0029 |
| Electronic | NDCG@100 | 0.0028 | 0.0029 | 0.0034 | 0.0900 | 0.0907 | 0.0034 |
|---------|--------|----------|----------|----------|----------|----------|----------|
| Health | NDCG@5 | 0.0155 | 0.0155 | 0.0155 | 0.0155 | 0.1060 | 0.0029 |
| Health | NDCG@10 | 0.0180 | 0.0180 | 0.0180 | 0.0180 | 0.1156 | 0.0045 |
| Health | NDCG@20 | 0.0219 | 0.0219 | 0.0219 | 0.0219 | 0.1375 | 0.0083 |
| Health | NDCG@50 | 0.0308 | 0.0308 | 0.0308 | 0.0308 | 0.1731 | 0.0251 |
| Health | NDCG@75 | 0.0367 | 0.0367 | 0.0367 | 0.0367 | 0.1902 | 0.0343 |
| Health | NDCG@100 | 0.0413 | 0.0413 | 0.0413 | 0.0413 | 0.2022 | 0.0434 |

## Notes
- SL@K: Results of SL@K method with different K values using LightGCN backbone
- This table extends Table 4 analysis to LightGCN backbone to test the generalizability across different K values
- Only Health and Electronic datasets are included (same as original Table 4)
- Results help understand how different K values affect performance on different backbone architectures
- SL@K indicates SL method trained with K value
