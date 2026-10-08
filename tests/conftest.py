"""Shared test setup: every test runs with the same fictional domain rules, never a local domain.yaml."""

import pytest

from kb.core.domain import Domain, use_domain

# The generic defaults plus a fictional product prompt ("ACME>", "<ACME>") used by the chunking tests.
TEST_DOMAIN = Domain(command_patterns=(r"<?ACME>",))


@pytest.fixture(autouse=True)
def _test_domain():
    """Use TEST_DOMAIN for the duration of each test."""
    use_domain(TEST_DOMAIN)
    yield
    use_domain(None)
