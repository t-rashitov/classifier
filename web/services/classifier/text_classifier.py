import logging
import os
import pathlib
import pickle
import re
from random import shuffle
from typing import Any

from pymorphy2 import MorphAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent

try:
    russian_stopwords = open(os.path.join(BASE_DIR, 'stopwords/russian.txt')).readlines()
except FileNotFoundError:
    russian_stopwords = []

morph_analyzer = MorphAnalyzer()


class NewsClassifier:
    """
    Класс для создания модели классификатора новостных материалов по рубрикам
    """

    def __init__(
            self,
            topic_files_names_with_titles: tuple,
            test_unparsed_news_file_name: str,
            test_parsed_news_file_name: str,
            test_titles_file_name: str,
            unparsed_data_path: str,
            parsed_data_path: str,
            test_data_path: str

    ):
        self.topic_files_names_with_titles = topic_files_names_with_titles
        self.test_unparsed_news_file_name = test_unparsed_news_file_name
        self.test_parsed_news_file_name = test_parsed_news_file_name
        self.test_titles_file_name = test_titles_file_name
        self.unparsed_data_path = unparsed_data_path
        self.parsed_data_path = parsed_data_path
        self.test_data_path = test_data_path

    @classmethod
    def get_default_classifier(cls):
        return cls(
            topic_files_names_with_titles=(
                ('politics.txt', 'политика'),
                ('society.txt', 'общество'),
                ('economy.txt', 'экономика'),
                ('sport.txt', 'спорт'),
                ('incidents.txt', 'проишествия'),
                ('culture.txt', 'культура'),
                ('technologies.txt', 'технологии'),
                ('science.txt', 'наука'),
                ('business.txt', 'бизнес'),
                ('health.txt', 'здоровье'),
                ('army.txt', 'армия'),
                ('games.txt', 'игры')
            ),
            test_unparsed_news_file_name='unparsed_test_news.txt',
            test_parsed_news_file_name='parsed_test_news.txt',
            test_titles_file_name='test_titles.txt',
            unparsed_data_path='data_unparsed',
            parsed_data_path='data_parsed',
            test_data_path='data_test'
        )

    @classmethod
    def parse_articles(cls, articles: list) -> list:
        """

        :param articles: список новостных статей
        :return parsed_articles: список предобработанных статей
        """

        parsed_articles = []

        for article in articles:
            article = article.lower().replace('ё', 'е')

            if 'http' in article:
                continue

            # article = re.sub(r'[^a-zа-я\s]', '', article)
            article = re.sub(r'[^a-zа-я ]', '', article)
            article = re.sub(r' +', ' ', article)
            article = article.lstrip()
            words = []

            for word in article.split(' '):

                if word not in russian_stopwords:

                    if len(word) > 1:
                        try:
                            words.append(morph_analyzer.parse(word)[0].normal_form)
                        except (AttributeError, IndexError):
                            logger.warning(f"Word <{word}> hasn't been changed.")
                            words.append(word)

                    else:
                        words.append(word)

            if len(words) < 10:
                continue

            words_line = ' '.join(words)

            parsed_articles.append(words_line + '\n')

        return parsed_articles

    @classmethod
    def parse_from_file(cls, input_file_path: str, output_file_path: str):
        """
        Метод для обработки текста статей из файла
        :param input_file_path: путь до входящего файла со статьями
        :param output_file_path: путь до исходящего файла со статьями
        :return:
        """

        file = open(input_file_path)

        try:
            articles = file.readlines()
        finally:
            file.close()

        logger.debug(f'articles count: {len(articles)}')

        parsed_articles = cls.parse_articles(articles)

        with open(output_file_path, 'w') as file:
            file.writelines(parsed_articles)

        logger.debug(f'parsed articles count: {len(parsed_articles)}')

    @classmethod
    def _get_articles_with_title(cls, path: str, title: str) -> []:
        """
        Метод для обработки и маппинга статей из текстового файла
        :param path: путь до файла со статьями
        :param title: заголовок для статей
        :return: []
        """

        file = open(path)

        try:
            return [(title, item) for item in file.readlines() if len(item) > 2]
        finally:
            file.close()

    @classmethod
    def load_object(cls, path: str) -> Any:
        """
        Метод десериализации объекта из файла
        :param path: Пусть до файла сериализованного объекта
        :return:
        """
        dumped_file = open(path, 'rb')

        try:
            deserialized_object = pickle.load(dumped_file)
        finally:
            dumped_file.close()

        return deserialized_object

    def _load_vectorizer(self) -> TfidfVectorizer:
        """
        Метод возвращает десериализованный объект векторизатора
        :return: объект TfidfVectorizer
        """

        return self.load_object(os.path.join(BASE_DIR, 'dumped_vectorizer.pkl'))

    def _load_classifier(self) -> RandomForestClassifier:
        """
        Метод возвращает десериализованный объект обученного классификатора
        :return: объект RandomForestClassifier
        """

        return self.load_object(os.path.join(BASE_DIR, 'dumped_classifier.pkl'))

    def _train_model(
            self,
            dump_vectorizer: bool = True,
            dump_classifier: bool = False,
            parse_files: bool = False
    ) -> RandomForestClassifier:
        """
        Метод обучения модели
        :param dump_vectorizer: определяет нужно ли сохранять вектор
        :param dump_classifier: определяет нужно ли сохранять обученный классификатор
        :param parse_files: определяется нужно ли обрабатывать файлы обучающего датасета
        :return: объект RandomForestClassifier
        """

        if parse_files:
            for file_name, _ in self.topic_files_names_with_titles:
                self.parse_from_file(
                    os.path.join(BASE_DIR, self.unparsed_data_path, file_name),
                    os.path.join(BASE_DIR, self.parsed_data_path, file_name)
                )

        articles = []
        for file_name, title in self.topic_files_names_with_titles:
            articles += self._get_articles_with_title(os.path.join(BASE_DIR, self.parsed_data_path, file_name), title)

        shuffle(articles)

        texts, titles = [], []

        for article in articles:
            titles.append(article[0])
            texts.append(article[1])

        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(texts)

        if dump_vectorizer:

            dump_file = open(os.path.join(BASE_DIR, 'dumped_vectorizer.pkl'), 'wb')

            try:
                pickle.dump(vectorizer, dump_file)
            finally:
                dump_file.close()

        training_data_vector = vectorizer.transform(texts)

        classifier = RandomForestClassifier(n_estimators=100, n_jobs=8)

        classifier.fit(training_data_vector, titles)

        texts, titles = [], []

        for article in articles:
            titles.append(article[0])
            texts.append(article[1])

        classifier.fit(training_data_vector, titles)

        if dump_classifier:

            dump_file = open(os.path.join(BASE_DIR, 'dumped_classifier.pkl'), 'wb')

            try:
                pickle.dump(classifier, dump_file)
            finally:
                dump_file.close()

        return classifier

    def get_predicted_category(self, articles: list = None, load_model: bool = False) -> list:
        """
        Классификация текста на основе обученной модели

        :param articles: новостные статьи для определения категорий
        :param load_model: определяет нужно ли загрузить сериализованную ранее модель
        :return: список предсказанных рубрик
        """

        if load_model:
            classifier = self._load_classifier()
        else:
            classifier = self._train_model(dump_vectorizer=True, dump_classifier=True, parse_files=True)

        if not articles:
            articles = open(os.path.join(BASE_DIR, self.test_data_path, self.test_parsed_news_file_name)).readlines()
        else:
            articles = self.parse_articles(articles)

        if not articles:
            return []

        vectorizer = self._load_vectorizer()

        data_vectors = vectorizer.transform(articles).toarray()

        return [classifier.predict([vector])[0] for vector in data_vectors]

    def test_predicted(self) -> None:
        """
        Метод для запуска тестирования классификатора на тестовом датасете
        :return:
        """

        self.parse_from_file(
            os.path.join(BASE_DIR, self.test_data_path, self.test_unparsed_news_file_name),
            os.path.join(BASE_DIR, self.test_data_path, self.test_parsed_news_file_name)
        )

        predicted_categories = self.get_predicted_category(load_model=True)

        file = open(os.path.join(BASE_DIR, self.test_data_path, self.test_titles_file_name))
        try:
            categories = file.readlines()
        finally:
            file.close()

        logger.debug(f'titles: {len(categories)} articles: {len(predicted_categories)}')

        if len(categories) == len(predicted_categories):

            counter = 0

            for i, category in enumerate(predicted_categories):
                logger.debug(f'{i}. Рубрика: {categories[i].rstrip()} -> классифицировано как: {category}')

                real_categories = categories[i].split()

                if category in real_categories:
                    counter += 1

            accuracy = counter / len(categories)

            logger.info(f'{accuracy=}')
