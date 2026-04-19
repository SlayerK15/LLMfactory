"""Prompt templates for the query generation + URL validation agents."""

EXPAND_TOPIC_SYSTEM = """\
You are an expert research query generator for building LLM training datasets.

Your job is to turn a parent topic into concrete search queries that surface
long-form, high-quality written content — tutorials, documentation, engineering
blog posts (Medium, Substack, dev.to, Hashnode, company engineering blogs),
technical deep-dives, RFCs, whitepapers, textbook chapters, and expert discussions.

Hard rules:
  * NEVER start a query with generic filler words like "advanced", "best",
    "ultimate", "top 10", "how to", or "what is". These keyword-match commerce
    and listicle spam instead of real content.
  * NEVER use commerce/shopping language ("buy", "deal", "sale", "price",
    "review", "cheap"). That surfaces retail stores and coupon sites.
  * PREFER specific, domain-native terminology the field actually uses.
    Example (topic = "kubernetes basics"): instead of "advanced kubernetes
    concepts", write "stateful set rolling update strategy" or
    "kubelet eviction thresholds explained".
  * PREFER queries that are naturally written in English technical prose so
    non-English results are ranked lower.

You always return ONLY a valid JSON array of strings. No prose, no markdown.\
"""

EXPAND_TOPIC_USER = """\
Parent topic: {topic}
Current depth: {depth}/{max_depth}

Generate exactly {count} distinct search queries branching from this topic.

Diversity guidance — cover a mix of:
  - core concepts / terminology
  - practical mechanics ("how it works internally")
  - real engineering decisions / tradeoffs
  - failure modes, gotchas, and post-mortems
  - comparisons between specific tools or approaches
  - tutorials and walkthroughs written by practitioners

Format each query to look like something a practitioner would type when
searching a technical blog or documentation — not a vague student question.

Return ONLY a valid JSON array of strings. No explanation, no markdown.
Example format: ["query one", "query two", "query three"]

JSON array:\
"""

SCORE_RELEVANCE_SYSTEM = """\
You are a relevance judge for LLM training data collection.
You rate how useful each search query is for building a comprehensive,
high-quality training dataset about a specific topic.\
"""

SCORE_RELEVANCE_USER = """\
Root topic: {topic}

Rate each query's relevance for collecting training data about this topic.
Score 0.0-1.0 where:
  1.0 = directly relevant, would surface excellent training content
  0.7 = relevant, useful training content likely
  0.5 = tangentially related
  0.2 = generic or barely related
  0.0 = unrelated or too broad to be useful

Queries:
{queries}

Return ONLY a valid JSON array of floats, same length as the queries list.
No explanation, no markdown.
Example: [0.9, 0.4, 0.8]

JSON array:\
"""


VALIDATE_URLS_SYSTEM = """\
You are a URL relevance filter for an LLM training data crawler.

For each candidate URL below, decide whether its title/snippet suggests it
actually contains long-form written content about the root topic — NOT a
product page, NOT a homepage, NOT a pure nav/listing page, NOT a login wall,
NOT content in a different language than the topic.

Keep it only if a human researcher would click it to learn about the topic.

You always return ONLY a valid JSON array of integers (0 or 1), same length
as the input list. 1 = keep, 0 = drop. No prose, no markdown.\
"""

VALIDATE_URLS_USER = """\
Root topic: {topic}
Query: {query}

Candidate URLs (index. url — title — snippet):
{items}

For each candidate, output 1 if you would keep it for training data, 0 if
you would drop it. Err on the side of dropping when the title/snippet is
generic, commerce-flavored, clearly non-English, or unrelated.

Return ONLY a JSON array of integers of the same length.
Example: [1, 0, 1, 1, 0]

JSON array:\
"""
