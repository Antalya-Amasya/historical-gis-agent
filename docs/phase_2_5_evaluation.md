# Phase 2.5 evaluation

Hash fallback Recall@5: 0.286.

Structure-aware plus semantic: Recall@1 0.571, Recall@3 0.857, Recall@5 0.857, MRR 0.690.
Fixed-v3 plus semantic: Recall@1 0.571, Recall@3 0.571, Recall@5 0.714, MRR 0.619.

Seven queries were manually annotated, including two Chinese queries. Both strategies retrieved the Chinese Alpine-difficulties query; only structure-aware retrieved the Chinese Livy query at Top-5. Both failed the English Livy route-into-Italy query. The set is too small to establish general superiority. In this set, structure-aware has higher Recall@3, Recall@5 and MRR. OCR and book/chapter detection quality in Livy remain limitations. Page classification excluded 43 explicit non-body fixed-window chunks; unknown pages remained included.
