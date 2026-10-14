"""Pure BM25 and rank-fusion tests requiring no private Vault or model."""
import pytest

from app.hybrid_retrieval import BM25Index, reciprocal_rank_fusion, tokenize


ROWS = [
    {'chunk_id': 1, 'title': '债券', 'heading': '票面利率', 'content': '债券票面利率确定每期利息', 'domain': 'finance'},
    {'chunk_id': 2, 'title': '货币政策', 'heading': '央行政策', 'content': '中央银行调整政策利率', 'domain': 'finance'},
    {'chunk_id': 3, 'title': '宋史', 'heading': '王安石变法', 'content': '北宋历史改革', 'domain': 'history'},
]


def test_tokenization_chinese_bigram_and_ascii():
    assert '债券' in tokenize('债券的票面利率')
    assert '利率' in tokenize('债券的票面利率')
    assert tokenize('Bond ETF 2026') == ['bond', 'etf', '2026']


def test_bm25_ranks_specific_finance_chunk_first():
    hits = BM25Index(ROWS).search('债券票面利率', top_k=3)
    assert hits[0][0]['chunk_id'] == 1
    assert hits[0][1] > 0


def test_bm25_domain_filter_and_unknown_query():
    index = BM25Index(ROWS)
    assert index.search('中央银行政策利率', domains=['history']) == []
    assert index.search('中央银行政策利率', domains=['finance'])[0][0]['chunk_id'] == 2
    assert index.search('xyzunseenphrase') == []


def test_rrf_rewards_consensus_and_stable_ties():
    dense = [(1, 'dense-first'), (2, 'dense-second')]
    lexical = [(2, 'lex-second'), (3, 'lex-third')]
    result = reciprocal_rank_fusion([dense, lexical], top_k=3)
    assert [x[0] for x in result] == [2, 1, 3]
    assert result[0][1] == 'dense-second'


def test_rrf_duplicate_ids_do_not_gain_extra_score():
    result = reciprocal_rank_fusion([[(1, 'first'), (1, 'duplicate'), (2, 'second')]], top_k=2)
    assert [x[0] for x in result] == [1, 2]
    assert result[0][2] == pytest.approx(1 / 61)


def test_empty_and_validation():
    assert BM25Index([]).search('anything') == []
    assert reciprocal_rank_fusion([[], []], top_k=4) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[]], top_k=0)
    with pytest.raises(ValueError):
        BM25Index(ROWS, k1=-1)
    with pytest.raises(ValueError):
        BM25Index(ROWS).search('test', top_k=0)
