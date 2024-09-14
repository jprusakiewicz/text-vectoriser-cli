import json

from pydantic import BaseModel, Field
from typing import Optional


class VectorizerConfig(BaseModel):
    stop_words: Optional[str] = Field(None,
                                      description="Specify 'english' to use a built-in list of stop words or a custom list of stop words.")
    lowercase: Optional[bool] = Field(True, description="Convert all characters to lowercase before tokenizing.")
    token_pattern: Optional[str] = Field(r'\b\w\w+\b', description="Regular expression for tokenizing strings.")
    ngram_range: Optional[tuple[int, int]] = Field((1, 1), description="Range of n-values for n-grams.")
    analyzer: Optional[str] = Field('word', description="Type of analyzer to use. Options are 'word' or 'char'.")
    binary: Optional[bool] = Field(False, description="Whether to return a binary matrix.")

    class Config:
        extra = 'forbid'


class CountVectorizerConfig(VectorizerConfig):
    max_df: Optional[float] = Field(1, description="Ignore terms with a document frequency higher than this.")
    min_df: Optional[int] = Field(1, description="Ignore terms with a document frequency lower than this.")
    max_features: Optional[int] = Field(None,
                                        description="Use only the top max_features ordered by term frequency across the corpus.")


class TfidfVectorizerConfig(VectorizerConfig):
    norm: Optional[str] = Field('l2', description="Normalization method. Options are 'l1', 'l2', or None.")
    smooth_idf: Optional[bool] = Field(True,
                                       description="Whether to add one to document frequencies to smooth idf weights.")
    sublinear_tf: Optional[bool] = Field(False, description="Whether to apply sublinear tf scaling.")
    use_idf: Optional[bool] = Field(True, description="Enable inverse-document-frequency reweighting.")


class HashingVectorizerConfig(VectorizerConfig):
    n_features: Optional[int] = Field(1048576, description="Number of features for the hashing space.")
    alternate_sign: Optional[bool] = Field(True, description="If True, hash values alternate signs.")


def build_vectorizer_config(vectorizer_type: str, config: str) -> VectorizerConfig:
    try:
        config = json.loads(config)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format.")

    match vectorizer_type:
        case 'count':
            return CountVectorizerConfig(**config)
        case 'tfidf':
            return TfidfVectorizerConfig(**config)
        case 'hashing':
            return HashingVectorizerConfig(**config)
        case _:
            raise ValueError("Unsupported vectorizer type.")
