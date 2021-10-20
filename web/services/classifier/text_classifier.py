import logging
import os
import pathlib
import pickle
import re
from random import shuffle
from typing import Any

import numpy as np
import pandas as pd
from pymorphy2 import MorphAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from services.classifier.exceptions import ModelFileError

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).parent

try:
    russian_stopwords = tuple(open(os.path.join(BASE_DIR, 'stopwords/russian.txt')).read().splitlines())
except FileNotFoundError:
    russian_stopwords = ()

morph_analyzer = MorphAnalyzer()


class NewsClassifier:
    """Класс для создания модели классификатора новостных материалов по рубрикам"""

    def __init__(
            self,
            topic_files_names_with_titles: tuple,
            test_unparsed_news_file_name: str,
            test_parsed_news_file_name: str,
            test_titles_file_name: str,
            unparsed_data_path: str,
            parsed_data_path: str,
            test_data_path: str,
            classifier: RandomForestClassifier = None,
            vectorizer: TfidfVectorizer = None

    ):
        self.topic_files_names_with_titles = topic_files_names_with_titles
        self.test_unparsed_news_file_name = test_unparsed_news_file_name
        self.test_parsed_news_file_name = test_parsed_news_file_name
        self.test_titles_file_name = test_titles_file_name
        self.unparsed_data_path = unparsed_data_path
        self.parsed_data_path = parsed_data_path
        self.test_data_path = test_data_path
        self.classifier = classifier
        self.vectorizer = vectorizer

    @classmethod
    def get_default_classifier(cls):
        instance = cls(
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
            test_data_path='data_test',
        )

        try:
            instance.classifier = cls._load_classifier()
            instance.vectorizer = cls._load_vectorizer()
        except FileNotFoundError:
            logger.debug('Model file not found. Training new model.')
            instance.train_model(dump_vectorizer=True, dump_classifier=True, parse_files=True)

        return instance

    @classmethod
    def _read_from_file(cls, file_path: str) -> list[str]:
        """
        Метод считывает список текстов из файла
        :param file_path: путь до файла с текстом
        :return: список текстов
        """

        with open(file_path) as file:
            return file.read().splitlines()

    @classmethod
    def _write_to_file(cls, texts: list[str], file_path: str, mode: str = 'w', separator: str = '\n') -> None:
        """
        Метод записывает список текстов в файл
        :param texts: список текстов
        :param file_path: пусть до файла
        :param mode: режим открытия файла
        :param separator: разделительный символ
        :return:
        """

        with open(file_path, mode) as file:
            file.write(separator.join(texts))

    @classmethod
    def parse(cls, articles: list) -> list:
        """
        Метод для обработки текстов
        :param articles: список новостных статей
        :return parsed_texts: список предобработанных текстов
        """

        parsed_texts = []

        for article in articles:

            article = re.sub(r'[^a-zа-яё ]', '', article.lower())
            article = re.sub(r' +', ' ', article)
            article = article.strip()
            words = []

            for word in article.split():

                if word not in russian_stopwords:

                    if len(word) > 1:
                        try:
                            words.append(morph_analyzer.parse(word)[0].normal_form)
                        except (AttributeError, IndexError) as e:
                            logger.error(str(e))
                            logger.debug(f'Word <{word}> hasn\'t been changed.')
                            words.append(word)

            if len(words) < 10:
                logger.debug('The number of words in the text is less than 10.')
                continue

            parsed_texts.append(' '.join(words))

        return parsed_texts

    @classmethod
    def parse_from_file(cls, input_file_path: str, output_file_path: str) -> None:
        """
        Метод для обработки текста статей из файла
        :param input_file_path: путь до входящего файла со статьями
        :param output_file_path: путь до исходящего файла со статьями
        :return:
        """

        articles = cls._read_from_file(input_file_path)

        logger.debug(f'Texts read from file: {len(articles)}')

        parsed_articles = cls.parse(articles)

        cls._write_to_file(texts=parsed_articles, file_path=output_file_path)

        logger.debug(f'Texts write to file: {len(parsed_articles)}')

    @classmethod
    def _get_articles_with_title(cls, path: str, title: str) -> list:
        """
        Метод для обработки и маппинга статей из текстового файла
        :param path: путь до файла со статьями
        :param title: заголовок для статей
        :return: список статей с заголовками
        """

        with open(path) as file:
            return [(title, item) for item in file.read().splitlines() if len(item) > 2]

    @classmethod
    def _load_object(cls, path: str) -> Any:
        """
        Метод десериализации объекта из файла
        :param path: Пусть до файла сериализованного объекта
        :return: десериализованный объект
        """

        with open(path, 'rb') as dumped_file:
            return pickle.load(dumped_file)

    @classmethod
    def _load_vectorizer(cls) -> TfidfVectorizer:
        """
        Метод возвращает десериализованный объект векторизатора
        :return: объект TfidfVectorizer
        """

        return cls._load_object(os.path.join(BASE_DIR, 'dumped_vectorizer.pkl'))

    @classmethod
    def _load_classifier(cls) -> RandomForestClassifier:
        """
        Метод возвращает десериализованный объект обученного классификатора
        :return: объект RandomForestClassifier
        """

        return cls._load_object(os.path.join(BASE_DIR, 'dumped_classifier.pkl'))

    def train_model(
            self,
            dump_vectorizer: bool = True,
            dump_classifier: bool = True,
            parse_files: bool = True
    ) -> None:
        """
        Метод обучения модели
        :param dump_vectorizer: определяет нужно ли сохранять вектор
        :param dump_classifier: определяет нужно ли сохранять обученный классификатор
        :param parse_files: определяется нужно ли обрабатывать файлы обучающего датасета
        :return:
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
            with open(os.path.join(BASE_DIR, 'dumped_vectorizer.pkl'), 'wb') as dump_file:
                pickle.dump(vectorizer, dump_file)

        training_data_vector = vectorizer.transform(texts)

        classifier = RandomForestClassifier(n_estimators=100, n_jobs=8)
        classifier.fit(training_data_vector, titles)

        texts, titles = [], []

        for article in articles:
            titles.append(article[0])
            texts.append(article[1])

        classifier.fit(training_data_vector, titles)

        if dump_classifier:
            with open(os.path.join(BASE_DIR, 'dumped_classifier.pkl'), 'wb') as dump_file:
                pickle.dump(classifier, dump_file)

        self.vectorizer = vectorizer
        self.classifier = classifier

    def get_predicted_category(self, articles: list) -> list:
        """
        Классификация текста на основе обученной модели

        :param articles: новостные статьи для определения категорий
        :return predicted: список предсказанных рубрик
        """

        articles = self.parse(articles)

        if not articles:
            return []

        if not (self.classifier and self.vectorizer):
            raise ModelFileError('Classifier model files does not exists. Run train_model() for create model.')

        data_vectors = self.vectorizer.transform(articles).toarray()

        vector_frame = pd.DataFrame(self.classifier.predict_proba(data_vectors), columns=self.classifier.classes_)
        values_frame = pd.DataFrame(vector_frame.columns.values[np.argsort(-vector_frame.values)[:, :3]])

        predicted = [(
                (row[0], vector_frame.iloc[i, vector_frame.columns.get_loc(row[0])]),
                (row[1], vector_frame.iloc[i, vector_frame.columns.get_loc(row[1])]),
                (row[2], vector_frame.iloc[i, vector_frame.columns.get_loc(row[2])])
            ) for i, row in enumerate(list(values_frame.itertuples(index=False, name=None)))
        ]

        return predicted

    def test_predicted(self) -> float:
        """
        Метод для запуска тестирования классификатора на тестовом датасете
        :return accuracy: точность классификации
        """

        self.parse_from_file(
            os.path.join(BASE_DIR, self.test_data_path, self.test_unparsed_news_file_name),
            os.path.join(BASE_DIR, self.test_data_path, self.test_parsed_news_file_name)
        )

        real_categories = self._read_from_file(os.path.join(BASE_DIR, self.test_data_path, self.test_titles_file_name))
        predicted_categories = self.get_predicted_category(
            articles=self._read_from_file(os.path.join(BASE_DIR, self.test_data_path, self.test_parsed_news_file_name)))

        logger.debug(f'Count of titles: {len(real_categories)}, articles: {len(predicted_categories)}')

        assert len(real_categories) == len(predicted_categories), \
            'The number of headings received does not match the number of controls.'

        correctly_predicted = [
            (real, predicted) for real, predicted in zip(real_categories, predicted_categories)
            if set(real.split()).intersection(set(dict(predicted).keys()))]

        accuracy = len(correctly_predicted) / len(real_categories)

        logger.debug('\n'.join(
            [f'{real}, classified as: {predicted}' for real, predicted in correctly_predicted]))
        logger.debug(f'{accuracy=}')

        return accuracy
