# Text Spam Detector

### This is a text spam detector created using a Bernoulli Naive Bayes Classifier integrated into a web application using Flask.

## Installation

CD into the backend directory, initialize environment and activate (file will be different depending on OS)

```bash
python -m venv myenv
myenv\Scripts\activate.bat
```

Install packages from requirements.txt

```bash
pip install -r requirements.txt
```

To export packages

```bash
pip freeze > requirements.txt
```

CD into the frontend directory and install packages

```bash
npm install
```

To run, CD into backend/flaskr and run
```bash
flask --app app.py run
```

To build frontend
```bash
npm run build
```
