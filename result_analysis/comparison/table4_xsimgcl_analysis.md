# Table 4 Extended: XSimGCL Backbone SL@K Performance Analysis

This table shows the performance of SL@K method with different K values using XSimGCL backbone on Health and Electronic datasets, extending the analysis beyond MF backbone.

| Dataset | Metric | SL@5 | SL@10 | SL@20 | SL@50 | SL@75 | SL@100 |
|---------|--------|----------|----------|----------|----------|----------|----------|
| Electronic | NDCG@5 | 0.0299 | 0.0306 | 0.0310 | 0.0312 | 0.0314 | 0.0321 |
| Electronic | NDCG@10 | 0.0372 | 0.0380 | 0.0383 | 0.0383 | 0.0384 | 0.0391 |
| Electronic | NDCG@20 | 0.0470 | 0.0480 | 0.0480 | 0.0482 | 0.0487 | 0.0489 |
| Electronic | NDCG@50 | 0.0628 | 0.0634 | 0.0638 | 0.0640 | 0.0641 | 0.0646 |
| Electronic | NDCG@75 | 0.0708 | 0.0714 | 0.0717 | 0.0719 | 0.0723 | 0.0727 |
| Electronic | NDCG@100 | 0.0769 | 0.0778 | 0.0779 | 0.0781 | 0.0785 | 0.0789 |
|---------|--------|----------|----------|----------|----------|----------|----------|
| Health | NDCG@5 | 0.0855 | 0.0892 | 0.0223 | 0.0787 | 0.0885 | 0.0883 |
| Health | NDCG@10 | 0.0972 | 0.1007 | 0.0245 | 0.0861 | 0.0983 | 0.0992 |
| Health | NDCG@20 | 0.1174 | 0.1206 | 0.0303 | 0.1041 | 0.1201 | 0.1197 |
| Health | NDCG@50 | 0.1537 | 0.1565 | 0.0428 | 0.1370 | 0.1555 | 0.1550 |
| Health | NDCG@75 | 0.1702 | 0.1726 | 0.0495 | 0.1524 | 0.1723 | 0.1720 |
| Health | NDCG@100 | 0.1820 | 0.1845 | 0.0562 | 0.1633 | 0.1848 | 0.1842 |

## Notes
- SL@K: Results of SL@K method with different K values using XSimGCL backbone
- This table extends Table 4 analysis to XSimGCL backbone to test the generalizability across different K values
- Only Health and Electronic datasets are included (same as original Table 4)
- Results help understand how different K values affect performance on different backbone architectures
- SL@K indicates SL method trained with K value
