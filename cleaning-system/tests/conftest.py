"""Shared fixtures for cleaning-system tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cleaning_system.core.models import CleaningConfig, DocRecord


def _make_doc(
    id: str,
    text: str,
    run_id: str = "test-run",
    url: str = "https://example.com",
    extraction_confidence: float = 0.95,
) -> DocRecord:
    import hashlib

    return DocRecord(
        id=id,
        run_id=run_id,
        url=url,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        token_count=int(len(text.split()) * 1.3),
        extraction_confidence=extraction_confidence,
        path=Path(f"/fake/{run_id}/docs/{id}.md"),
    )


# ---------------------------------------------------------------------------
# Sample docs — each is ~80-120 words, clearly English, clearly about DevOps.
# Word count is above the Gopher min_words=50 threshold by design.
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_docs() -> list[DocRecord]:
    """Ten realistic English docs covering DevOps themes (80-120 words each)."""
    return [
        _make_doc(
            "doc1",
            "Continuous integration and continuous deployment are the backbone of modern "
            "software development practices. Teams that adopt CI/CD pipelines release software "
            "faster and with fewer bugs compared to teams without automation. Jenkins, GitHub "
            "Actions, and GitLab CI are popular choices for automating build, test, and deploy "
            "workflows. Each commit triggers an automated pipeline that runs unit tests, "
            "integration tests, and security scans before merging code to the main branch. "
            "This shift-left approach catches defects early when they are cheapest to fix.",
        ),
        _make_doc(
            "doc2",
            "Docker containers package applications and their dependencies into portable "
            "immutable units that run consistently across development, staging, and production. "
            "Kubernetes orchestrates these containers at scale, handling scheduling, health "
            "checks, rolling updates, and automatic restarts. The combination of Docker and "
            "Kubernetes has become the de facto standard for deploying microservices in cloud "
            "environments. Engineers write Helm charts to define their application deployments "
            "as version-controlled code, enabling repeatable and auditable releases that can "
            "be rolled back instantly if a problem is detected after deployment.",
        ),
        _make_doc(
            "doc3",
            "Infrastructure as Code tools like Terraform and Ansible enable teams to manage "
            "cloud resources declaratively in version-controlled configuration files. Instead "
            "of manual console clicks that cannot be audited or reproduced, engineers define "
            "every virtual machine, network, and database as code. Terraform plans show exactly "
            "what will change before any resource is modified, reducing the risk of accidental "
            "destructive operations in production. Ansible playbooks configure running servers "
            "idempotently, ensuring that repeated runs produce the same result regardless of "
            "the current state of the target machine.",
        ),
        _make_doc(
            "doc4",
            "Monitoring and observability are critical for maintaining reliable distributed "
            "systems. Prometheus collects metrics from instrumented services and stores them "
            "as time-series data. Grafana visualises those metrics on dashboards that give "
            "on-call engineers instant insight into system health. Loki aggregates logs from "
            "every container and indexes them for fast querying alongside metrics. Setting up "
            "alerting rules in Alertmanager ensures that engineers are notified when error "
            "rates exceed thresholds or latency spikes before customers notice degradation. "
            "Mean time to recovery improves significantly when teams invest in solid "
            "observability tooling from the beginning of a project.",
        ),
        _make_doc(
            "doc5",
            "Site reliability engineering bridges the gap between software development and "
            "IT operations by applying software engineering principles to infrastructure "
            "problems. SRE teams define service level objectives and error budgets that "
            "quantify reliability targets in measurable terms. When the error budget is "
            "exhausted, feature work slows down and reliability improvements take priority. "
            "Google popularised SRE through their published book, which has influenced "
            "engineering culture across the entire industry. Toil elimination, blameless "
            "postmortems, and capacity planning are core SRE practices that improve both "
            "system reliability and team morale over the long term.",
        ),
        _make_doc(
            "doc6",
            "Git is the standard version control system used by nearly every software team "
            "worldwide. Branching strategies like GitFlow and trunk-based development help "
            "teams collaborate without stepping on each other's changes. Trunk-based "
            "development keeps the main branch always releasable by requiring short-lived "
            "feature branches and frequent integration. Pull request reviews with automated "
            "test gates, linting checks, and security scanning enforce code quality standards "
            "before any change lands in the main branch. Semantic versioning and automated "
            "changelog generation make release management predictable and transparent for "
            "all stakeholders who depend on the software.",
        ),
        _make_doc(
            "doc7",
            "Secret management is a critical security concern in modern DevOps pipelines. "
            "Hardcoding credentials in source code or environment variables exposes "
            "organisations to significant risk of data breaches. Tools like HashiCorp Vault "
            "and AWS Secrets Manager provide centralised, audited access to sensitive "
            "credentials, API keys, and certificates. Dynamic secrets that expire after a "
            "short time window limit the blast radius of any compromise. Rotating secrets "
            "regularly and auditing access logs with automated anomaly detection further "
            "reduces security risk in cloud-native deployments that use ephemeral compute "
            "resources across multiple regions and availability zones.",
        ),
        _make_doc(
            "doc8",
            "Blue-green and canary deployment strategies minimise downtime and risk when "
            "shipping new software versions to production. Blue-green deployments maintain "
            "two identical environments and switch traffic instantly between them, enabling "
            "zero-downtime deployments with a simple DNS change. Canary releases gradually "
            "shift a small percentage of production traffic to the new version while "
            "monitoring error rates and latency. If metrics remain healthy, traffic shifts "
            "continue until the new version handles one hundred percent of requests. Feature "
            "flags allow teams to decouple deployment from release, enabling dark launches "
            "and instant rollbacks without code changes.",
        ),
        _make_doc(
            "doc9",
            "Platform engineering teams build internal developer platforms that abstract "
            "away infrastructure complexity behind self-service interfaces. Golden paths, "
            "service catalogues, and automated scaffolding let application developers "
            "deploy confidently without deep knowledge of Kubernetes internals or cloud "
            "provider networking. Backstage, the open-source portal from Spotify, has "
            "become a popular foundation for internal developer platforms. Teams define "
            "templates that encode organisational best practices so that new services "
            "start correctly configured from day one. Reducing cognitive load on developers "
            "by abstracting platform concerns into reusable components accelerates delivery "
            "velocity across the entire engineering organisation.",
        ),
        _make_doc(
            "doc10",
            "Chaos engineering deliberately injects failures into production systems to "
            "verify that services handle disruption gracefully and recover automatically. "
            "Netflix Chaos Monkey randomly terminates virtual machine instances to ensure "
            "that no single instance failure causes customer-facing outages. Running regular "
            "game days where teams deliberately break things in controlled ways builds "
            "confidence in system resilience and uncovers hidden single points of failure. "
            "Chaos experiments should be hypothesis-driven, starting with small blast "
            "radius and gradually increasing scope as confidence grows. The goal is not "
            "to cause incidents but to discover failure modes before real incidents expose "
            "them unexpectedly to customers.",
        ),
    ]


@pytest.fixture
def cleaning_config(tmp_path: Path) -> CleaningConfig:
    return CleaningConfig(
        run_id="test-run",
        topic="DevOps",
        data_dir=tmp_path,
        enable_perplexity=False,
        enable_relevance=False,
        enable_trafilatura=False,
    )


@pytest.fixture
def populated_data_dir(tmp_path: Path, sample_docs: list[DocRecord]) -> Path:
    """Write sample_docs to the filesystem layout expected by collection-system."""
    docs_dir = tmp_path / "runs" / "test-run" / "docs"
    docs_dir.mkdir(parents=True)

    for doc in sample_docs:
        (docs_dir / f"{doc.id}.md").write_text(doc.text, encoding="utf-8")
        (docs_dir / f"{doc.id}.meta.json").write_text(
            json.dumps(
                {
                    "id": doc.id,
                    "run_id": doc.run_id,
                    "url": doc.url,
                    "title": doc.title,
                    "content_hash": doc.content_hash,
                    "token_count": doc.token_count,
                    "extraction_confidence": doc.extraction_confidence,
                }
            ),
            encoding="utf-8",
        )

    return tmp_path
