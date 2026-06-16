# Mixed Correctness Results

- model: `qwen2.5:14b`
- base_url: `http://202.120.24.199:13000/v1`
- n_medqa: 100
- n_medsafety: 100

| Metric | Value |
| --- | ---: |
| MedQA correctness | 0.760 |
| MedSafety correctness | 0.840 |
| Mixed correctness macro | 0.800 |
| Pooled correctness | 0.800 |
| MedQA abstain rate | 0.020 |
| MedSafety refusal/redirect rate | 0.840 |

## Route Counts

- `ACCURACY`: 69
- `DYNAMIC_MDT`: 98
- `ERROR`: 1
- `SAFETY_REVIEW_HIGH`: 15
- `SAFETY_REVIEW_LOW`: 17
