# -*- coding: utf-8 -*-
from flask import request, jsonify

from services.classifier.text_classifier import NewsClassifier
from classifier import app

news_classifier = NewsClassifier.get_default_classifier()


@app.route('/', methods=('POST',))
def predict_view():
    if request.method == 'POST':

        data = request.get_json()
        articles = data.get('articles')

        predicted = news_classifier.get_predicted_category(articles=articles)

        return jsonify(dict(predicted=predicted))
