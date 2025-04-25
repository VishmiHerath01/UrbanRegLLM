# Answer Quality Metrics Summary

Total Queries Analyzed: 10
- With Reference Answers: 10
- Without Reference Answers: 0

## Direct Comparison (All Queries)

### Answer Length

- Regular Embeddings: 1196.2 characters
- MRL Embeddings: 1357.7 characters
- Difference: 161.5 characters (13.5%)

### Answer Similarity

Average semantic similarity between regular and MRL answers: 0.0000

## Reference-Based Metrics

| Metric | Regular Embeddings | MRL Embeddings | Difference | % Improvement |
|--------|-------------------|----------------|------------|---------------|
| ROUGE1 | 0.2962 | 0.2812 | -0.0150 | -5.05% |
| ROUGE2 | 0.1225 | 0.1196 | -0.0029 | -2.39% |
| ROUGEL | 0.2010 | 0.2040 | 0.0030 | 1.48% |
| BLEU | 0.0663 | 0.0569 | -0.0094 | -14.11% |
| BERT_F1 | 0.0000 | 0.0000 | 0.0000 | 0.00% |

### Overall Comparison

MRL embeddings performed better in 1 out of 5 metrics.

### Per-Query Analysis

Number of queries where MRL embeddings performed better:

- ROUGE1: 4/10 queries (40.00%)
- ROUGE2: 7/10 queries (70.00%)
- ROUGEL: 6/10 queries (60.00%)
- BLEU: 6/10 queries (60.00%)
- BERT_F1: 0/10 queries (0.00%)
