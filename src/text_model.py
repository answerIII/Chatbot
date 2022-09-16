import os
import re
import json
import string
import random
import pymorphy2

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score

from model_wrappers import Word2VecWrapper
from utils import json_reader, preproces_text, clean_json


class TextModel:
    def __init__(self, dir_path, filename, vectorizer_type, analyzer_fg):
        self.dir_path = dir_path
        self.filename = filename.split('.')[0]
        self.vectorizer_type = vectorizer_type
        self.analyzer_fg = analyzer_fg

        file_path = os.path.join(dir_path, filename)
        self.data = json_reader(file_path)

        self.vectorizer_model = None
        self.model = None

        self.model_path = os.path.join(os.curdir, '../models', 'my_model')
        if os.path.exists(os.path.join(self.model_path)):
            self.model = tf.keras.models.load_model('../models/my_model')

        self.run_model_pipeline(analyzer_fg=self.analyzer_fg)

    @staticmethod
    def preprocessing_text(text):
        text = text.lower()
        # patterns = r"[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        # patterns = r"[0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
        # text = re.sub(patterns, '', text)

        tokens = word_tokenize(text)
        tokens = [i for i in tokens if i != '']
        # tokens = [i for i in tokens if (i not in string.punctuation)]

        # stop_words = stopwords.words('russian', 'english')
        # stop_words.extend(['a', 'я', 'ты', 'вы', 'это', 'этот', 'тот', 'моей'])
        # tokens = [i for i in tokens if (i not in stop_words)]

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(i) for i in tokens]

        # morph = pymorphy2.MorphAnalyzer()
        # tokens = [morph.parse(token)[0].normal_form for token in tokens]
        return tokens

    def prepare_train_data(self):
        doc_X = []
        doc_y = []
        classes = {}
        for intent_name, intent_data in self.data['intents'].items():
            for pattern in intent_data['examples']:
                tokens = self.preprocessing_text(pattern)
                if tokens:
                    doc_X.append(' '.join(tokens))
                    doc_y.append(intent_name)
                    if intent_name not in classes:
                        classes[intent_name] = 0
                    classes[intent_name] += 1
        return doc_X, doc_y

    def vectorizer(self, X):
        if self.vectorizer_type == 'BOW':
            # params: ngram_range=(2, 4)
            self.vectorizer_model = CountVectorizer()
        elif self.vectorizer_type == 'TFIDF':
            # analyzer='char_wb', ngram_range=(2, 4)
            self.vectorizer_model = TfidfVectorizer()
        elif self.vectorizer_type == 'W2V':
            self.vectorizer_model = Word2VecWrapper()
            self.vectorizer_model.fit(X)
        X_vec = self.vectorizer_model.fit_transform(X)
        return X_vec

    def predictor(self, X, y):

        knn = KNeighborsClassifier().fit(X, y)
        print('Theoretical score Knn:', knn.score(X, y))

        linearsvc = LinearSVC(random_state=42, tol=1e-5).fit(X, y)
        print('Theoretical score LinearSVC:', linearsvc.score(X, y))

        self.model = knn

    def ml_analyzer(self, X, y):
        linear_svc = LinearSVC(random_state=42, tol=1e-5)

        # cross validation split (stratify=y)
        scores = []
        for i in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = linear_svc.fit(X_train, y_train)
            scores.append(self.model.score(X_test, y_test))
        print(f'CV by train_test_split')
        print(f'scores: {scores}')
        print(f'mean score: {sum(scores)/len(scores)}')

        # cross validation score
        val_score = cross_val_score(linear_svc, X, y, scoring='accuracy', cv=5)
        print(f'CV by cross_val_score')
        print(f'scores: {val_score}')
        print(f'mean score: {sum(val_score)/len(val_score)}')

    def run_model_pipeline(self, analyzer_fg):
        X, y = self.prepare_train_data()
        X_vec = self.vectorizer(X)
        self.predictor(X_vec, y)
        if analyzer_fg:
            self.ml_analyzer(X_vec, y)

    def get_response(self, text):
        text = self.preprocessing_text(text)
        text_vec = self.vectorizer_model.transform([' '.join(text)])
        intent = self.model.predict(text_vec.toarray())
        return random.choice(self.data['intents'][intent[0]]['examples'])

    def console_test(self):
        message = ''
        stop_message = 'пока-пока'
        while message != stop_message:
            message = input('')
            message = preproces_text(message)
            answer = self.get_response(message)
            print(answer)


if __name__ == '__main__':

    filepath = f'../data'
    source_filename = f'data.json'
    text_model = TextModel(filepath, source_filename, vectorizer_type='BOW', analyzer_fg=False)
    text_model.console_test()

