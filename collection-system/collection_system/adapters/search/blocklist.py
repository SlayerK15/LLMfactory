"""
Shared URL/domain blocklist for all search adapters.

The goal is to drop results that are almost never useful LLM training content:
commercial/retail surfaces, non-English regional portals, generic support pages,
social-media account pages, app store listings, etc.

`is_blocked(domain, url)` is the single entry point — adapters call it once per
candidate URL and drop anything that matches. High-signal blog platforms
(medium.com, substack.com, dev.to, hashnode.dev, ...) are intentionally NOT
blocked so they flow through to the validator.
"""
from __future__ import annotations

# Exact-match full domain (also applies to any subdomain). e.g. "amazon.com"
# matches amazon.com AND www.amazon.com AND shop.amazon.com.
BLOCKED_EXACT: frozenset[str] = frozenset(
    [
        # --- social / platform noise ---
        "pinterest.com", "facebook.com", "twitter.com", "x.com",
        "instagram.com", "tiktok.com", "snapchat.com", "threads.net",
        "linkedin.com",

        # --- commerce / retail (these hijack every "advanced X" query) ---
        "amazon.com", "amazon.in", "amazon.co.uk", "amazon.de",
        "ebay.com", "ebay.co.uk",
        "walmart.com", "target.com", "bestbuy.com", "costco.com",
        "alibaba.com", "aliexpress.com", "temu.com", "shein.com",
        "etsy.com", "wayfair.com", "homedepot.com", "lowes.com",
        "advanceautoparts.com", "autozone.com", "oreillyauto.com",
        "napaonline.com", "rockauto.com", "carid.com",

        # --- app stores & download sites ---
        "apps.apple.com", "play.google.com", "microsoft.com/store",
        "apps.microsoft.com", "softonic.com", "en.softonic.com",
        "filehippo.com", "softpedia.com", "cnet.com",

        # --- dictionary / generic reference (not useful for technical topics) ---
        "merriam-webster.com", "dictionary.com", "thesaurus.com",
        "vocabulary.com", "wordreference.com",

        # --- Chinese / regional portals that never return English content ---
        "zhihu.com", "baidu.com", "weibo.com", "sina.com.cn",
        "csdn.net", "jianshu.com", "tianya.cn", "douban.com",
        "bilibili.com", "youku.com", "toutiao.com", "36kr.com",
        "qq.com",  # includes pvp.qq.com and friends
    ]
)

# Any domain ending with one of these suffixes is blocked.
BLOCKED_SUFFIX: tuple[str, ...] = (
    # Chinese/qq subdomains
    ".qq.com", ".baidu.com", ".zhihu.com", ".sina.com.cn",
    # Indonesian / regional government procurement (pure noise)
    ".go.id",
    # Retail subdomains
    ".amazon.com", ".amazon.in", ".amazon.co.uk", ".ebay.com",
    ".advanceautoparts.com", ".autozone.com",
)

# Any domain starting with one of these prefixes is blocked.
# Use sparingly — very easy to over-match.
BLOCKED_PREFIX: tuple[str, ...] = (
    "shop.",          # shop.advanceautoparts.com, shop.xxx.com (commerce)
    "store.",         # online-store subdomains
    "buy.",           # retail
    "cart.",          # retail
    "checkout.",      # retail
    "m.facebook.com", # mobile social
)

# Substrings in the URL path that signal junk we don't want.
# Checked against the FULL URL (lower-cased), not just the domain.
BLOCKED_URL_SUBSTR: tuple[str, ...] = (
    "/cart/", "/checkout/", "/my-account",
    "msockid=",   # Microsoft tracking ID — pure shopping-surface spam
    "/signin",    # login walls
    "/login",
)

# Forum-style Q&A sites that are generally low-quality for pretraining
# (off-topic, conversational, often in the wrong language). We leave
# stackoverflow + unix.stackexchange + serverfault alone because tech content
# there is solid.
BLOCKED_QA_FORUMS: frozenset[str] = frozenset(
    [
        "forums.studentdoctor.net",
        "ell.stackexchange.com",      # English-learner Q&A
        "math.stackexchange.com",     # general math, usually off-topic
        "forum.wordreference.com",
        "quora.com",
        "answers.yahoo.com",
        "yahoo.com",
    ]
)


def _domain_matches(domain: str, exact: frozenset[str]) -> bool:
    """True if `domain` is exactly in `exact` or is a subdomain of one."""
    if domain in exact:
        return True
    for e in exact:
        if domain.endswith("." + e):
            return True
    return False


def is_blocked(domain: str, url: str | None = None) -> bool:
    """
    Return True if the given domain (and optional URL) should be dropped.

    Callers pass both so we can match on commerce query-strings (msockid=…)
    as well as domain patterns. Domain matching handles exact-equals,
    subdomain-of-exact, suffix patterns, and prefix patterns.
    """
    d = (domain or "").lower().strip()
    if not d:
        return True  # malformed — drop

    if _domain_matches(d, BLOCKED_EXACT):
        return True
    if _domain_matches(d, BLOCKED_QA_FORUMS):
        return True
    for suf in BLOCKED_SUFFIX:
        if d.endswith(suf):
            return True
    for pre in BLOCKED_PREFIX:
        if d.startswith(pre):
            return True

    if url:
        u = url.lower()
        for s in BLOCKED_URL_SUBSTR:
            if s in u:
                return True
    return False


# Backwards-compat alias so existing imports keep working.
BLOCKED_DOMAINS = BLOCKED_EXACT
