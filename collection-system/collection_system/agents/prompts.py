"""Prompt templates for the query generation agent."""

EXPAND_TOPIC_SYSTEM = """\
You are an expert research query generator for building LLM training datasets.
Given a parent topic or sub-topic, generate specific search queries that would
surface high-quality, diverse training content about that subject.
Focus on queries that would retrieve actual web pages, articles, tutorials,
documentation, and expert discussions — not definitions or list pages.\
"""

EXPAND_TOPIC_USER = """\
Parent topic: {topic}
Current depth: {depth}/{max_depth}

Generate exactly {count} distinct search queries branching from this topic.
Cover different angles such as: fundamentals, advanced concepts, practical
applications, real-world examples, comparisons, best practices, and common pitfalls.

Return ONLY a valid JSON array of strings. No explanation, no markdown.
Example: ["query one", "query two", "query three"]

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
Score 0.0–1.0 where:
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
