# SuisenSha Recommender Benchmark Report

- **Generated**: 2026-10-11 12:59:36 UTC
- **Evaluation Protocol**: Leave-two-out sequential split (validation held-out, test isolated)
- **Seeds Evaluated**: 5 ([42, 43, 44, 45, 46])
- **Dataset**: MovieLens-100K (938 users, 1,682 items)

## 1. Ranking Accuracy (Mean ± 95% CI)

| Model | HitRate@5 | HitRate@10 | NDCG@5 | NDCG@10 | Recall@10 | MRR@10 | Train Time (s) | Peak Mem (MB) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Random | 0.0034 ± 0.0025 | 0.0077 ± 0.0033 | 0.0021 ± 0.0018 | 0.0034 ± 0.0021 | 0.0077 ± 0.0033 | 0.0022 ± 0.0018 | 0.01s | 0.2 MB |
| Popularity | 0.0469 | 0.0810 | 0.0297 ± 0.0000 | 0.0406 | 0.0810 | 0.0285 | 0.01s | 0.1 MB |
| ItemKNN | 0.0757 | 0.1151 ± 0.0000 | 0.0522 | 0.0651 | 0.1151 ± 0.0000 | 0.0498 | 15.17s | 35.7 MB |
| BPR-MF | 0.0682 ± 0.0021 | 0.1087 ± 0.0063 | 0.0432 ± 0.0036 | 0.0562 ± 0.0049 | 0.1087 ± 0.0063 | 0.0403 ± 0.0048 | 9.17s | 22.7 MB |
| **SASRec** | 0.0708 ± 0.0078 | 0.1213 ± 0.0061 | 0.0471 ± 0.0030 | 0.0633 ± 0.0019 | 0.1213 ± 0.0061 | 0.0460 ± 0.0013 | 424.36s | 4.3 MB |

## 2. Beyond-Accuracy & Exposure Metrics (Mean ± 95% CI)

| Model | Coverage@10 | Diversity@10 | Novelty@10 | Avg Popularity@10 | Gini Exposure@10 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 0.8476 ± 0.0011 | 0.8242 ± 0.0026 | 6.2404 ± 0.0409 | 35.0396 ± 0.8943 | 0.3393 ± 0.0038 |
| Popularity | 0.0380 | 0.7803 | 1.5833 | 318.7068 | 0.9893 ± 0.0000 |
| ItemKNN | 0.1225 | 0.8025 | 2.0608 | 243.5942 | 0.9657 |
| BPR-MF | 0.1495 ± 0.0078 | 0.7996 ± 0.0189 | 2.1702 ± 0.0159 | 236.2564 ± 2.8936 | 0.9627 ± 0.0022 |
| **SASRec** | 0.3866 ± 0.0805 | 0.8027 ± 0.0070 | 3.0567 ± 0.1805 | 148.6460 ± 12.4189 | 0.8572 ± 0.0302 |

## 3. Performance by User Activity Quartile (HitRate@10)

| Model | Q1 (Least Active) | Q2 | Q3 | Q4 (Most Active) |
|:---|:---:|:---:|:---:|:---:|
| Random | 0.0094 ± 0.0079 | 0.0068 ± 0.0060 | 0.0077 ± 0.0044 | 0.0068 ± 0.0103 |
| Popularity | 0.1068 | 0.0468 | 0.1068 | 0.0638 |
| ItemKNN | 0.2094 | 0.1149 ± 0.0000 | 0.0940 | 0.0426 |
| BPR-MF | 0.1761 ± 0.0233 | 0.1047 ± 0.0157 | 0.0932 ± 0.0121 | 0.0613 ± 0.0060 |
| **SASRec** | 0.2085 ± 0.0069 | 0.1166 ± 0.0080 | 0.0855 ± 0.0263 | 0.0749 ± 0.0161 |

## 4. Performance by Item Popularity Tier

| Model | Head HR@10 | Torso HR@10 | Tail HR@10 | Head Exposure Share | Tail Exposure Share |
|:---|:---:|:---:|:---:|:---:|:---:|
| Random | 0.0088 ± 0.0027 | 0.0066 ± 0.0082 | 0.0027 ± 0.0075 | 0.2146 ± 0.0052 | 0.4286 ± 0.0076 |
| Popularity | 0.1282 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| ItemKNN | 0.1821 | 0.0000 | 0.0000 | 0.9945 ± 0.0000 | 0.0021 |
| BPR-MF | 0.1710 ± 0.0099 | 0.0015 ± 0.0025 | 0.0027 ± 0.0075 | 0.9772 ± 0.0059 | 0.0063 ± 0.0013 |
| **SASRec** | 0.1771 ± 0.0051 | 0.0325 ± 0.0114 | 0.0000 | 0.8363 ± 0.0415 | 0.0115 ± 0.0082 |

## 5. Explanation Quality Evaluation

| Metric | Result |
|:---|:---:|
| Evaluated Explanations | 25 |
| Grounding Rate | 100.0% |
| Hallucination Rate | 4.0% |
| LLM-as-Judge Relevance | 4.00 / 5.0 |
| Average Length (words) | 13.0 words |
| Average Length (characters) | 83.6 chars |
| Average Latency | 0.05 ms |
