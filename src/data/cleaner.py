"""Privacy-preserving text normalization and cleaning utilities."""
import re
from typing import Optional


class TextCleaner:
    """Cleans customer and brand tweet text, masking PII and non-informative tokens."""

    # Regex patterns for privacy masking
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
    PHONE_PATTERN = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
    TWITTER_HANDLE_PATTERN = re.compile(r"@\w+")
    WHITESPACE_PATTERN = re.compile(r"\s+")

    @classmethod
    def clean(
        cls,
        text: Optional[str],
        mask_urls: bool = True,
        mask_handles: bool = True,
        mask_pii: bool = True,
    ) -> str:
        """Clean and normalize a text string."""
        if not text or not isinstance(text, str):
            return ""

        # Normalize line endings
        cleaned = text.replace("\r", " ").replace("\n", " ")

        # Mask URLs
        if mask_urls:
            cleaned = cls.URL_PATTERN.sub("<URL>", cleaned)

        # Mask Emails & Phone numbers
        if mask_pii:
            cleaned = cls.EMAIL_PATTERN.sub("<EMAIL>", cleaned)
            cleaned = cls.PHONE_PATTERN.sub("<PHONE>", cleaned)

        # Mask Twitter handles
        if mask_handles:
            # Mask user handles to @user, preserve brand mentions if needed
            cleaned = cls.TWITTER_HANDLE_PATTERN.sub("@user", cleaned)

        # Normalize multiple spaces
        cleaned = cls.WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    @classmethod
    def is_english_text(cls, text: str, threshold: float = 0.85) -> bool:
        """Heuristic check to ensure text is primarily ASCII/English."""
        if not text or not isinstance(text, str):
            return False
        total = len(text)
        if total == 0:
            return False
        ascii_count = sum(1 for c in text if ord(c) < 128)
        return (ascii_count / total) >= threshold
