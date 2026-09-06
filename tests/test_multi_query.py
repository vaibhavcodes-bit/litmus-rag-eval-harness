from unittest.mock import patch

from langchain_core.documents import Document

from src.retrieval.multi_query import (
    deduplicate_documents,
    generate_queries,
    intent_phrase_score,
    lexical_relevance_score,
    multi_query_retrieve,
    rank_documents,
    rerank_by_original_question,
    weighted_rank_documents,
)

def make_document(
    content: str,
    chunk_id: str,
    source: str = "test.pdf",
):
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
        },
    )


def test_deduplicate_documents_removes_duplicate_chunk_ids():
    documents = [
        make_document("Chunk A", "chunk_1"),
        make_document("Chunk B", "chunk_2"),
        make_document("Chunk B duplicate", "chunk_2"),
        make_document("Chunk C", "chunk_3"),
    ]

    result = deduplicate_documents(documents)

    chunk_ids = [
        document.metadata["chunk_id"]
        for document in result
    ]

    assert chunk_ids == [
        "chunk_1",
        "chunk_2",
        "chunk_3",
    ]


def test_deduplicate_documents_preserves_order():
    documents = [
        make_document("Chunk C", "chunk_3"),
        make_document("Chunk A", "chunk_1"),
        make_document("Chunk C duplicate", "chunk_3"),
        make_document("Chunk B", "chunk_2"),
    ]

    result = deduplicate_documents(documents)

    chunk_ids = [
        document.metadata["chunk_id"]
        for document in result
    ]

    assert chunk_ids == [
        "chunk_3",
        "chunk_1",
        "chunk_2",
    ]


def test_generate_queries_returns_requested_number():
    mock_response = type(
        "MockResponse",
        (),
        {
            "content": (
                "first search query\n"
                "second search query\n"
                "third search query\n"
                "fourth search query"
            )
        },
    )()

    with patch(
        "src.retrieval.multi_query.get_query_generator"
    ) as mock_get_llm:

        mock_llm = mock_get_llm.return_value
        mock_llm.invoke.return_value = mock_response

        queries = generate_queries(
            "What is the first step in a job search?",
            query_count=4,
        )

    assert len(queries) == 4
    assert all(isinstance(query, str) for query in queries)
    assert all(query.strip() for query in queries)


def test_generate_queries_removes_duplicate_queries():
    mock_response = type(
        "MockResponse",
        (),
        {
            "content": (
                "same query\n"
                "same query\n"
                "different query\n"
                "another query"
            )
        },
    )()

    with patch(
        "src.retrieval.multi_query.get_query_generator"
    ) as mock_get_llm:

        mock_llm = mock_get_llm.return_value
        mock_llm.invoke.return_value = mock_response

        queries = generate_queries(
            "Test question",
            query_count=4,
        )

    assert queries == [
        "same query",
        "different query",
        "another query",
        "Test question",
    ]

def test_generate_queries_rejects_empty_question():
    try:
        generate_queries("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_multi_query_retrieve_merges_and_deduplicates():
    documents_by_query = {
        "original question": [
            make_document("Original Chunk", "chunk_0"),
            make_document("Chunk A", "chunk_1"),
        ],
        "query one": [
            make_document("Chunk A", "chunk_1"),
            make_document("Chunk B", "chunk_2"),
        ],
        "query two": [
            make_document("Chunk B", "chunk_2"),
            make_document("Chunk C", "chunk_3"),
        ],
        "query three": [
            make_document("Chunk C", "chunk_3"),
            make_document("Chunk D", "chunk_4"),
        ],
    }

    with patch(
        "src.retrieval.multi_query.generate_queries"
    ) as mock_generate_queries:

        mock_generate_queries.return_value = [
            "query one",
            "query two",
            "query three",
        ]

        with patch(
            "src.retrieval.multi_query.retrieve_documents"
        ) as mock_retrieve:

            def retrieve_side_effect(question, k):
                return documents_by_query[question]

            mock_retrieve.side_effect = retrieve_side_effect

            result = multi_query_retrieve(
                "original question",
                query_count=3,
                k=10,
            )

    chunk_ids = [
        document.metadata["chunk_id"]
        for document in result
    ]

    # All unique chunks must be present.
    assert set(chunk_ids) == {
        "chunk_0",
        "chunk_1",
        "chunk_2",
        "chunk_3",
        "chunk_4",
    }

    # Every chunk must appear only once.
    assert len(chunk_ids) == 5

    # The original query result must be included.
    assert "chunk_0" in chunk_ids

    # The repeated chunks must also be included.
    assert "chunk_1" in chunk_ids
    assert "chunk_2" in chunk_ids
    assert "chunk_3" in chunk_ids

def test_multi_query_retrieve_rejects_empty_question():
    try:
        multi_query_retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_rank_documents_prioritizes_documents_retrieved_by_multiple_queries():
    doc_a = Document(
        page_content="Document A",
        metadata={"chunk_id": "a"},
    )

    doc_b = Document(
        page_content="Document B",
        metadata={"chunk_id": "b"},
    )

    doc_c = Document(
        page_content="Document C",
        metadata={"chunk_id": "c"},
    )

    documents_by_query = [
        [doc_a, doc_b],
        [doc_b, doc_c],
        [doc_b],
    ]

    ranked = rank_documents(documents_by_query)

    assert ranked[0].metadata["chunk_id"] == "b"


def test_rank_documents_preserves_first_seen_order_for_ties():
    doc_a = Document(
        page_content="Document A",
        metadata={"chunk_id": "a"},
    )

    doc_b = Document(
        page_content="Document B",
        metadata={"chunk_id": "b"},
    )

    documents_by_query = [
        [doc_a],
        [doc_b],
    ]

    ranked = rank_documents(documents_by_query)

    assert ranked[0].metadata["chunk_id"] == "a"
    assert ranked[1].metadata["chunk_id"] == "b"
    
    
    
def test_rank_documents_uses_rank_position():
    doc_a = Document(
        page_content="Document A",
        metadata={"chunk_id": "chunk_a"},
    )

    doc_b = Document(
        page_content="Document B",
        metadata={"chunk_id": "chunk_b"},
    )

    doc_c = Document(
        page_content="Document C",
        metadata={"chunk_id": "chunk_c"},
    )

    documents_by_query = [
        [doc_a, doc_b, doc_c],
        [doc_b, doc_a, doc_c],
    ]

    ranked = rank_documents(documents_by_query)

    ranked_ids = [
        document.metadata["chunk_id"]
        for document in ranked
    ]

    assert ranked_ids[0] in {"chunk_a", "chunk_b"}
    assert ranked_ids[1] in {"chunk_a", "chunk_b"}
    assert ranked_ids[2] == "chunk_c"
    
    
def test_rerank_by_original_question_returns_documents():
    documents = [
        Document(
            page_content="Hidden job market networking referrals.",
            metadata={"chunk_id": "wrong"},
        ),
        Document(
            page_content=(
                "First step: begin with self-assessment, "
                "clarify career goals, target roles, "
                "target industries, and non-negotiables."
            ),
            metadata={"chunk_id": "correct"},
        ),
    ]

    class FakeEmbeddings:
        def embed_query(self, question):
            return [1.0, 0.0]

        def embed_documents(self, texts):
            return [
                [0.0, 1.0],
                [1.0, 0.0],
            ]

    with patch(
        "src.retrieval.multi_query.get_embeddings",
        return_value=FakeEmbeddings(),
    ):
        ranked = rerank_by_original_question(
            documents,
            "What is the recommended first step?",
        )

    assert ranked[0].metadata["chunk_id"] == "correct"
    assert ranked[1].metadata["chunk_id"] == "wrong"
    
    
    
    
def test_weighted_rank_documents_prioritizes_original_question():
    original_doc = Document(
        page_content="First step: begin with self-assessment.",
        metadata={"chunk_id": "correct"},
    )

    repeated_doc = Document(
        page_content="Hidden job market and networking.",
        metadata={"chunk_id": "wrong"},
    )

    # These are the results from the ORIGINAL question.
    original_documents = [
        original_doc,
        repeated_doc,
    ]

    # These are results ONLY from generated queries.
    #
    # The original query results are intentionally NOT included here.
    documents_by_query = [
        [
            repeated_doc,
        ],
        [
            repeated_doc,
        ],
    ]

    ranked = weighted_rank_documents(
        original_documents=original_documents,
        documents_by_query=documents_by_query,
        original_question=(
            "What is the recommended first step "
            "before starting a job search?"
        ),
        original_weight=2.0,
        lexical_weight=2.0,
    )

    assert ranked[0].metadata["chunk_id"] == "correct"
    
    
    
def test_lexical_relevance_score_prefers_matching_document():
    question = (
        "What is the recommended first step "
        "before starting a job search?"
    )

    correct_doc = Document(
        page_content=(
            "First step: begin with self-assessment. "
            "Clarify career goals, target roles, "
            "target industries, and non-negotiables "
            "before applying."
        ),
        metadata={"chunk_id": "correct"},
    )

    unrelated_doc = Document(
        page_content=(
            "The hidden job market refers to openings "
            "filled through networking and referrals."
        ),
        metadata={"chunk_id": "wrong"},
    )

    correct_score = lexical_relevance_score(
        question,
        correct_doc,
    )

    unrelated_score = lexical_relevance_score(
        question,
        unrelated_doc,
    )

    assert correct_score > unrelated_score 
    
def test_intent_phrase_score_prefers_correct_document():
    question = (
        "What is the recommended first step "
        "before starting a job search?"
    )

    correct_doc = Document(
        page_content=(
            "First step: begin with self-assessment. "
            "Clarify career goals, target roles, "
            "target industries, and non-negotiables "
            "before applying."
        ),
        metadata={"chunk_id": "correct"},
    )

    generic_doc = Document(
        page_content=(
            "The hidden job market refers to openings "
            "filled through networking, referrals, "
            "or internal promotion."
        ),
        metadata={"chunk_id": "generic"},
    )

    correct_score = intent_phrase_score(
        question,
        correct_doc,
    )

    generic_score = intent_phrase_score(
        question,
        generic_doc,
    )

    assert correct_score > generic_score  