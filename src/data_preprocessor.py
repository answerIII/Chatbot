import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class DataPreprocessor:
    def __init__(self):
        self.tokens = None
        self.stop_words = stopwords.words('russian')

    def run_pipeline(self, corpus):
        # self.tokens = self.delete_signs(self.delete_stop_words(self.tokenize(corpus)))
        self.tokens = self.tokenize(corpus)
        return self.tokens

    def tokenize(self, corpus):
        tokens = word_tokenize(corpus)
        return tokens

    def delete_stop_words(self, tokens):
        tokens = [i for i in tokens if (i not in self.stop_words)]
        return tokens

    def delete_signs(self, tokens):
        tokens = [i.replace("«", "").replace("»", "") for i in tokens]
        return tokens

    def lemma_rus(self, tokens):
        morph = pymorphy2.MorphAnalyzer()
        tokens = [morph.parse(token)[0].normal_form for token in tokens]
        return tokens
