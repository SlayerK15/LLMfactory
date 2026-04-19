from enum import StrEnum


class Stage(StrEnum):
    QUERY_GENERATION = "QUERY_GENERATION"
    URL_DISCOVERY = "URL_DISCOVERY"
    URL_VALIDATION = "URL_VALIDATION"
    SCRAPE = "SCRAPE"
    FINALIZE = "FINALIZE"


class RunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PAUSED = "PAUSED"


class URLStatus(StrEnum):
    PENDING = "PENDING"
    SCRAPED = "SCRAPED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class SearchBackend(StrEnum):
    CC_CDX = "CC_CDX"
    SEARXNG = "SEARXNG"
    DDG_LITE = "DDG_LITE"


class TrainingStyle(StrEnum):
    PRETRAIN = "pretrain"
    INSTRUCT = "instruct"


CHECKPOINT_INTERVAL_DOCS = 100
URL_HASH_ALGORITHM = "sha256"
CONTENT_HASH_ALGORITHM = "sha256"
