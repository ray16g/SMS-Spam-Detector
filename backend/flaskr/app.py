from flask import Flask, request, render_template
from classifier import classifyText

app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='static')

@app.route("/")
def main():
    return render_template('index.html', name=__name__)

@app.get("/api/classify")
def classify():
    return classifyText(request.args.get('data'))
