import pytest


@pytest.fixture
def test_cleaner():
    pass


def test_cleaner_with_empty_string():
    assert cleaner("") == ""


def test_cleaner_with_string_with_spaces():
    assert cleaner(" ") == ""