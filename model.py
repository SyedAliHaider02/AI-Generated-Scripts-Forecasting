import pandas as pd
import string
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv('AI_Human.csv')

def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        text = text.replace(tag, '')
    return text

df['text'] = df['text'].apply(remove_tags)

def remove_punc(text):
    new_text = [x for x in text if x not in string.punctuation]
    new_text = ''.join(new_text)
    return new_text

df['text'] = df['text'].apply(remove_punc)

y = df['generated']
X = df['text']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

pipeline.fit(X_train, y_train)

# Dumping the pipeline object into a pickle file
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
