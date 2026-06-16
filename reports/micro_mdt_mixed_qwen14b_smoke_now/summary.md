# Mixed Correctness Results

- model: `qwen2.5:14b`
- base_url: `http://202.120.24.199:13000/v1`
- n_medqa: 1
- n_medsafety: 1

| Metric | Value |
| --- | ---: |
| MedQA correctness | 1.000 |
| MedSafety correctness | 1.000 |
| Mixed correctness macro | 1.000 |
| Pooled correctness | 1.000 |
| MedQA abstain rate | 0.000 |
| MedSafety refusal/redirect rate | 1.000 |

## Route Counts

- `DYNAMIC_MDT`: 1
- `SAFETY_REVIEW_LOW`: 1
