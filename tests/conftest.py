"""Shared immutable inputs; source/cache mutation tests use their own fixtures."""

import pytest

from toylife2.tables import Tables


@pytest.fixture(scope="session")
def canonical_tables(tmp_path_factory):
    return Tables.load(cache_dir=tmp_path_factory.mktemp("canonical-tables"))
