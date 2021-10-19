import pytest

from services.classifier.text_classifier import NewsClassifier


@pytest.fixture(scope='session')
def classifier() -> 'NewsClassifier':
    return NewsClassifier.get_default_classifier()

