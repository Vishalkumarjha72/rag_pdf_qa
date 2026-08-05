import unittest

from app.retrieval import _reciprocal_rank_fusion


def _chunk(source: str, chunk_index: int) -> dict:
    return {
        "text": f"chunk {chunk_index}",
        "source": source,
        "page": 1,
        "score": 0.0,
        "metadata": {"chunk_index": chunk_index, "section_title": "s", "document_title": "d", "chunk_length": 1},
    }


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_chunk_ranked_first_in_both_lists_wins(self):
        dense = [_chunk("doc", 0), _chunk("doc", 1), _chunk("doc", 2)]
        keyword = [_chunk("doc", 0), _chunk("doc", 3), _chunk("doc", 1)]

        fused = _reciprocal_rank_fusion([dense, keyword])

        self.assertEqual(fused[0]["metadata"]["chunk_index"], 0)

    def test_chunk_appearing_in_both_lists_outranks_single_list_top_pick(self):
        # chunk 1 is rank-2 in both lists; chunk 2 is rank-1 in only the dense list.
        dense = [_chunk("doc", 2), _chunk("doc", 1)]
        keyword = [_chunk("doc", 3), _chunk("doc", 1)]

        fused = _reciprocal_rank_fusion([dense, keyword])
        fused_indices = [c["metadata"]["chunk_index"] for c in fused]

        self.assertEqual(fused_indices[0], 1)

    def test_no_duplicate_chunks_in_fused_output(self):
        dense = [_chunk("doc", 0), _chunk("doc", 1)]
        keyword = [_chunk("doc", 1), _chunk("doc", 0)]

        fused = _reciprocal_rank_fusion([dense, keyword])

        self.assertEqual(len(fused), 2)

    def test_empty_lists_produce_empty_result(self):
        self.assertEqual(_reciprocal_rank_fusion([[], []]), [])


if __name__ == "__main__":
    unittest.main()
