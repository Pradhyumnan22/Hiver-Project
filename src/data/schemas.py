"""Pydantic schemas for raw tweets, cleaned tweets, and linked conversation pairs."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field


class RawTweet(BaseModel):
    """Schema representing an individual row in twcs.csv."""
    tweet_id: int
    author_id: str
    inbound: bool
    created_at: str
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[float] = None


class ConversationPair(BaseModel):
    """Schema representing a linked customer query and corresponding brand response."""
    pair_id: str = Field(description="Unique identifier for the conversation pair, e.g. brand_id-cust_id")
    customer_tweet_id: int
    customer_author_id: str
    customer_text: str
    cleaned_customer_text: str
    brand_tweet_id: int
    brand_author_id: str
    brand_text: str
    cleaned_brand_text: str
    created_at_customer: Optional[str] = None
    created_at_brand: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class DatasetPartition(BaseModel):
    """Container for dev and evaluation dataset splits with provenance metadata."""
    brand: str
    created_timestamp: str
    total_pairs: int
    dev_pairs_count: int
    eval_pairs_count: int
    dev_pairs: List[ConversationPair]
    eval_pairs: List[ConversationPair]
