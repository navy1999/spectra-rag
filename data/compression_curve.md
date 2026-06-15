# Embedding compression curve (PCA)

Corpus: 282 node embeddings, 1024-d (Jina). recall@10 measured against the full-dim cosine neighbours. "Whitened" = Euclidean in PCA-whitened space (= Mahalanobis in the original space).

| dims (K) | bytes/node | compression | variance kept | recall@10 (PCA cosine) | recall@10 (PCA whitened) |
|---|---|---|---|---|---|
| 1024 (full) | 4096 | 1.0x | 100% | 1.000 | 1.000 |
| 8 | 32 | 128x | 32% | 0.426 | 0.412 |
| 16 | 64 | 64x | 46% | 0.572 | 0.538 |
| 32 | 128 | 32x | 64% | 0.721 | 0.620 |
| 64 | 256 | 16x | 82% | 0.817 | 0.598 |
| 128 | 512 | 8x | 96% | 0.851 | 0.451 |
| 256 | 1024 | 4x | 100% | 0.846 | 0.131 |
