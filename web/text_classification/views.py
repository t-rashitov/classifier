# -*- coding: utf-8 -*-
from flask import request, jsonify

from services.classifier.text_classifier import NewsClassifier
from text_classification import app

news_classifier = NewsClassifier.get_default_classifier()


@app.route('/', methods=('POST',))
def predict():
    if request.method == 'POST':

        data = request.get_json()

        articles = data.get('articles')

        # region Read from request file

        # file = request.files.get('news_file')
        #
        # if not file:
        #     return Response(status=400)
        #
        # lines = file.stream.readlines()
        #
        # articles = [line.decode('utf-8') for line in lines]

        # endregion

        predicted = news_classifier.get_predicted_category(articles=articles, load_model=True)

        return jsonify(dict(predicted=predicted))
