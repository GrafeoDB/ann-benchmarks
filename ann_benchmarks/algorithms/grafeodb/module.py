import numpy as np

from ..base.module import BaseANN


class GrafeoHNSW(BaseANN):
    def __init__(self, metric, method_param):
        self._metric = {"angular": "cosine", "euclidean": "euclidean"}[metric]
        self._m = method_param["M"]
        self._ef_construction = method_param["efConstruction"]
        self._ef = 50
        self.name = "grafeodb (%s)" % method_param

    def fit(self, X):
        from grafeo import GrafeoDB

        n, d = X.shape
        self._db = GrafeoDB()

        # Bulk-insert vectors as node properties (single FFI call).
        self._db.batch_create_nodes("V", "embedding", X.tolist())

        # Build the HNSW index on the stored vectors
        self._db.create_vector_index(
            "V",
            "embedding",
            dimensions=d,
            metric=self._metric,
            m=self._m,
            ef_construction=self._ef_construction,
        )

    def set_query_arguments(self, ef):
        self._ef = ef
        self.name = "grafeodb (M=%d, efC=%d, ef=%d)" % (
            self._m,
            self._ef_construction,
            ef,
        )

    def query(self, v, n):
        results = self._db.vector_search(
            "V", "embedding", v.tolist(), k=n, ef=self._ef
        )
        return [node_id for node_id, _ in results]

    def batch_query(self, X, n):
        results = self._db.batch_vector_search(
            "V", "embedding", X.tolist(), k=n, ef=self._ef
        )
        self.res = [[node_id for node_id, _ in r] for r in results]

    def freeIndex(self):
        del self._db
