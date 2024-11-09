from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import FunctionTransformer

from src.features.tfm import flatten_string_array_col, todense, tokenizer


def tags_pipeline_steps(count_vect_kwargs: dict):
    steps = [
        (
            "flatten_string_array_col",
            FunctionTransformer(flatten_string_array_col, validate=False),
        ),
        (
            "count_vect",
            CountVectorizer(
                tokenizer=tokenizer, token_pattern=None, **count_vect_kwargs
            ),
        ),
        ("todense", FunctionTransformer(todense, validate=False)),
    ]
    return steps
