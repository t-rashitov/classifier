import os

import pytest

from ..text_classifier import NewsClassifier, BASE_DIR


@pytest.fixture
def classifier() -> 'NewsClassifier':
    return NewsClassifier.get_default_classifier()


@pytest.mark.files
@pytest.mark.test_files
def test_classifier_testing_files_exists(classifier):
    """
    Проверка существования файлов для тестирования
    """

    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_unparsed_news_file_name))
    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_parsed_news_file_name))
    assert os.path.isfile(os.path.join(BASE_DIR, classifier.test_data_path, classifier.test_titles_file_name))


@pytest.mark.files
@pytest.mark.dump_files
def test_classifier_dumps_files_exists():
    """
    Провека существования файлов дампа модели и индекса классификатора
    """

    assert os.path.isfile(os.path.join(BASE_DIR, 'dumped_classifier.pkl'))
    assert os.path.isfile(os.path.join(BASE_DIR, 'dumped_vectorizer.pkl'))


@pytest.mark.files
@pytest.mark.stopwords_files
def test_stopwords_files_exists():
    """
    Проверка существования файлов стопслов
    """

    assert os.path.isfile(os.path.join(BASE_DIR, 'stopwords/russian.txt'))


@pytest.mark.files
@pytest.mark.learn_files
def test_default_classifier_learn_files_exists(classifier):
    """
    Проверка существования файлов для обучения модели
    """

    for name, _ in classifier.topic_files_names_with_titles:
        assert os.path.isfile(os.path.join(BASE_DIR, classifier.unparsed_data_path, name))
        assert os.path.isfile(os.path.join(BASE_DIR, classifier.parsed_data_path, name))


@pytest.mark.parser
def test_parser_http_in_text(classifier):
    """
    Проверка парсера (исключение статей со словом http)
    """

    text = 'test ' * 10 + 'http'
    assert len(classifier.parse_articles([text])) == 0


@pytest.mark.parser
def test_parser_text_replacing(classifier):
    """
    Проверка парсера (удаление невалидных символов)
    """

    long_string = 'test1234567890-+!@#$%^&*(){}[],./~`"№;:? тест Ё ' * 10
    parsed_strings = classifier.parse_articles([long_string])

    assert len(parsed_strings) == 1

    assert parsed_strings[0] == ('test тест е ' * 10)


@pytest.mark.parser
def test_parser_from_file(classifier):
    """
    Проверка парсера (парсинг статей из файла)
    """
    lines = [' '.join([title] * 10 + ['\n']) for _, title in classifier.topic_files_names_with_titles]

    try:
        with open('test_unparsed_articles.txt', 'w') as file:
            file.writelines(lines)

        classifier.parse_from_file('test_unparsed_articles.txt', 'test_parsed_articles.txt')

        file = open('test_parsed_articles.txt')
        articles = file.readlines()

    finally:
        os.remove('test_unparsed_articles.txt')
        os.remove('test_parsed_articles.txt')

    assert len(articles) == len(classifier.topic_files_names_with_titles)
