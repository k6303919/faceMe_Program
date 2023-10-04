
import cv2
from .routes import stream
from flask import Flask, request, render_template, Blueprint
from flask_cors import CORS

# app
app = Flask(__name__)
app.register_blueprint(stream.app, url_prefix='/stream')


def start():
    print("stream server start")
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
