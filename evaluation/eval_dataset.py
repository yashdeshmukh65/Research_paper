"""
Ground truth evaluation dataset for ANNA.

Each entry contains:
- question         : the user query
- relevant_pages   : dict of {paper_name: [page_numbers]} that should be retrieved
- expected_keywords: keywords that a correct answer must contain (case-insensitive)
- reference_answer : a reference answer for faithfulness/relevance scoring
- category         : type of question for stratified analysis
"""

EVAL_DATASET = [
    {
        "id": "Q01",
        "question": "What deep learning model architectures are proposed in the uploaded papers?",
        "relevant_pages": {},          # populated at runtime from indexed papers
        "expected_keywords": ["model", "architecture", "layer", "network", "proposed"],
        "reference_answer": "The papers propose various deep learning architectures including neural network models with multiple layers.",
        "category": "methodology"
    },
    {
        "id": "Q02",
        "question": "What datasets were used to evaluate the models?",
        "relevant_pages": {},
        "expected_keywords": ["dataset", "training", "test", "benchmark", "evaluation"],
        "reference_answer": "The models were evaluated on standard benchmark datasets.",
        "category": "dataset"
    },
    {
        "id": "Q03",
        "question": "What are the main limitations mentioned by the authors?",
        "relevant_pages": {},
        "expected_keywords": ["limitation", "drawback", "constraint", "challenge", "future"],
        "reference_answer": "The authors mention limitations related to data requirements, computational cost, and generalization.",
        "category": "limitations"
    },
    {
        "id": "Q04",
        "question": "What evaluation metrics are reported in the experiments?",
        "relevant_pages": {},
        "expected_keywords": ["accuracy", "precision", "recall", "f1", "metric", "performance"],
        "reference_answer": "The papers report standard evaluation metrics such as accuracy, precision, recall, and F1-score.",
        "category": "evaluation"
    },
    {
        "id": "Q05",
        "question": "What future research directions do the authors suggest?",
        "relevant_pages": {},
        "expected_keywords": ["future", "work", "direction", "propose", "extend", "improve"],
        "reference_answer": "The authors suggest future work including improved data efficiency and extended model generalization.",
        "category": "future_work"
    },
    {
        "id": "Q06",
        "question": "How do the proposed methods compare to baseline approaches?",
        "relevant_pages": {},
        "expected_keywords": ["baseline", "comparison", "outperform", "improve", "result"],
        "reference_answer": "The proposed methods outperform baseline approaches on the reported benchmarks.",
        "category": "results"
    },
    {
        "id": "Q07",
        "question": "What is the research problem being addressed in these papers?",
        "relevant_pages": {},
        "expected_keywords": ["problem", "challenge", "address", "objective", "goal"],
        "reference_answer": "The papers address core research problems in their respective domains.",
        "category": "research_problem"
    },
    {
        "id": "Q08",
        "question": "What training procedures or optimization strategies are described?",
        "relevant_pages": {},
        "expected_keywords": ["train", "optim", "loss", "epoch", "learning rate", "gradient"],
        "reference_answer": "The papers describe training procedures including optimization strategies and hyperparameter settings.",
        "category": "methodology"
    },
]

# Manually labelled limitation statements for precision/recall/F1 evaluation
# Format: {paper_name: [list of ground truth limitation sentences]}
LIMITATION_GROUND_TRUTH = {
    # These will be populated by the user with actual paper-specific limitations
    # Example structure shown below — replace with real paper content
    "__example__": [
        "The model requires large amounts of labelled training data.",
        "Computational requirements are high during training.",
        "The approach does not generalize well to out-of-distribution samples.",
        "Interpretability of the model decisions is limited.",
    ]
}

# Chunk size configurations for Experiment 3
CHUNK_SIZE_CONFIGS = [
    {"chunk_size_tokens": 500,  "overlap_tokens": 50},
    {"chunk_size_tokens": 800,  "overlap_tokens": 100},
    {"chunk_size_tokens": 1000, "overlap_tokens": 150},   # current default
    {"chunk_size_tokens": 1500, "overlap_tokens": 200},
]

# Embedding models for Experiment 4
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",          # current default — fast, small
    "all-mpnet-base-v2",          # larger, more accurate
    "paraphrase-MiniLM-L3-v2",    # smallest, fastest
]
