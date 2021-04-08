import logging

from pymorphy2 import MorphAnalyzer


logger = logging.getLogger(__name__)


def find_news_by_word(path: str, search_words: [], stop_words: [] = None, match_rate: float = 1):
    """
    Функция для поиска вхождения слов в статьи
    :param path: путь до файла со статьями
    :param search_words: список, искомых в статьях, слов
    :param stop_words: список слов которые не должны встречаться в статьях
    :param match_rate: коэффициент совпадения слов (слов найдено в тексте / всего слов)
    :return:
    """

    file = open(path)
    articles = file.readlines()
    file.close()

    logger.info(f'articles on enter: {len(articles)}')

    morph_analyzer = MorphAnalyzer()

    # region Обработка ключевых слов

    prepared_search_words = []
    prepared_stop_words = []

    if stop_words is None:
        stop_words = []

    for word in search_words:

        try:
            prepared_search_words.append(morph_analyzer.parse(word)[0].normal_form)
        except (AttributeError, IndexError):
            logger.warning(f"Word <{word}> hasn't been changed.")
            prepared_search_words.append(word)

    for word in stop_words:

        try:
            prepared_stop_words.append(morph_analyzer.parse(word)[0].normal_form)
        except (AttributeError, IndexError):
            logger.warning(f"Word <{word}> hasn't been changed.")
            prepared_stop_words.append(word)

    # endregion

    prepared_search_words_count = len(prepared_search_words)

    valid_articles = []
    valid_articles_counter = 0

    for article in articles:
        is_suitable = True

        for word in prepared_stop_words:

            if word in article:
                is_suitable = False
                break

        if is_suitable:

            number_of_matches_found = 0
            article_words = article.split(' ')

            for word in prepared_search_words:

                if word in article_words:
                    number_of_matches_found += 1

            if (number_of_matches_found / prepared_search_words_count) >= match_rate:
                valid_articles.append(article)
                valid_articles_counter += 1
                continue

    logger.info(f'valid articles: {valid_articles_counter}')

    try:
        file = open('valid_articles_output.txt', 'w')
        file.writelines(valid_articles)

    finally:
        file.close()
