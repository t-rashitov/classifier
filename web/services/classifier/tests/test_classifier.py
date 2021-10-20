import os

import pytest

from services.classifier.text_classifier import BASE_DIR


@pytest.mark.files
@pytest.mark.test_files
def test_unparsed_news_file_exists(classifier):
    """Проверка существования файла тестовых текстов новостей (необработанных)"""

    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_unparsed_news_file_name))


@pytest.mark.files
@pytest.mark.test_files
def test_parsed_news_file_exists(classifier):
    """Проверка существования файла тестовых текстов новостей (обработанных)"""

    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_parsed_news_file_name))


@pytest.mark.files
@pytest.mark.test_files
def test_titles_file_exists(classifier):
    """Проверка существования файла тестовых заголовков"""

    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_titles_file_name))


@pytest.mark.files
@pytest.mark.dump_files
def test_dumped_classifier_file_exists():
    """Провека существования файлов дампа модели классификатора"""

    assert os.path.isfile(os.path.join(BASE_DIR, 'dumped_classifier.joblib'))


@pytest.mark.files
@pytest.mark.dump_files
def test_dumped_vectorizer_file_exists():
    """Провека существования файлов дампа индекса классификатора"""

    assert os.path.isfile(os.path.join(BASE_DIR, 'dumped_vectorizer.joblib'))


@pytest.mark.files
@pytest.mark.stopwords_files
def test_stopwords_files_exists():
    """Проверка существования файлов стопслов"""

    assert os.path.isfile(os.path.join(BASE_DIR, 'stopwords/russian.txt'))


@pytest.mark.files
@pytest.mark.learn_files
def test_default_classifier_unparsed_data_files_exists(classifier):
    """Проверка существования файлов для обучения модели (необработанных)"""

    for name, _ in classifier.topic_files_names_with_titles:
        assert os.path.isfile(os.path.join(BASE_DIR, classifier.unparsed_data_path, name))


@pytest.mark.files
@pytest.mark.learn_files
def test_default_classifier_parsed_data_files_exists(classifier):
    """Проверка существования файлов для обучения модели (обработанных)"""

    for name, _ in classifier.topic_files_names_with_titles:
        assert os.path.isfile(os.path.join(BASE_DIR, classifier.parsed_data_path, name))


@pytest.mark.parser
def test_parser_text_replacing(classifier):
    """Проверка парсера (удаление невалидных символов)"""

    long_string = 'test1234567890-+!@#$%^&*(){}[],./~`"№;:? тест ЁёЁ ' * 10
    parsed_strings = classifier.parse([long_string])

    assert len(parsed_strings) == 1

    assert parsed_strings[0] == ' '.join(['test тест ёёё' for _ in range(10)])


@pytest.mark.parser
def test_parser_from_file(classifier):
    """Проверка парсера (парсинг статей из файла)"""

    lines = [' '.join([title for _ in range(10)]) for _, title in classifier.topic_files_names_with_titles]

    try:
        classifier._write_to_file(texts=lines, file_path='test_unparsed_articles.txt')
        classifier.parse_from_file('test_unparsed_articles.txt', 'test_parsed_articles.txt')
        articles = classifier._read_from_file('test_parsed_articles.txt')

    finally:
        os.remove('test_unparsed_articles.txt')
        os.remove('test_parsed_articles.txt')

    assert len(articles) == len(classifier.topic_files_names_with_titles)
