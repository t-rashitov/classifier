from flask import Flask

app = Flask(__name__)

from classifier import views
