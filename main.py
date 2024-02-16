from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import pickle
import string

app = Flask(__name__)
app.secret_key = 'b83a1e0ea4e74d22c5d6a3a0ff5e6e66'

# Load the trained model
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)


class Form(FlaskForm):
    text = StringField('Enter the text', validators=[DataRequired()])
    submit = SubmitField('Submit')


@app.route('/', methods=['GET', 'POST'])
def home():
    form = Form()
    Predictions = None
    if form.validate_on_submit():
        text = form.text.data
        Predictions = Predictionss(text)
    return render_template('home.html', form=form, prediction=Predictions)


def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text


def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text


def Predictionss(text):
    text = remove_tags(text)
    text = remove_punc(text)
    pred = pipeline.predict([text])[0]
    return pred


if __name__ == '__main__':
    app.run(debug=True)
