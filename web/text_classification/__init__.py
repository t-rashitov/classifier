from flask import Flask


app = Flask(__name__)


from text_classification import views
