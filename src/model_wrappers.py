import numpy as np
from tqdm import tqdm
from os.path import exists
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder

from data_preprocessor import DataPreprocessor


class Word2VecWrapper:

    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model = None

    def fit(self, data):
        sentences = self.data_preparing(data)
        if exists('../models/word2vec.model'):
            print(f'W2V used pretrained')
            self.model = Word2Vec.load('../models/word2vec.model')
        else:
            self.model = Word2Vec(sentences, vector_size=500, window=5, min_count=5, epochs=10)
            self.model.save('../models/word2vec.model')

    def transform(self, data):
        sentences = self.data_preparing(data)
        embedded_sentences = np.array([])
        for sentence in tqdm(sentences):
            vector = [self.model.wv[word] for word in sentence
                      if word in self.model.wv.key_to_index.keys()]
            vector = np.mean(vector, axis=0) if vector else np.zeros(500)
            embedded_sentences = vector if embedded_sentences.size == 0 else np.vstack((embedded_sentences, vector))
        return embedded_sentences

    def data_preparing(self, data):
        return [self.data_preprocessor.run_pipeline(sent) for sent in data]


class OHEWrapper:
    def __init__(self):
        self.model = OneHotEncoder(handle_unknown='ignore')

    def fit(self, data):
        self.model.fit(data.values.reshape(-1, 1))

    def transform(self, data):
        return self.model.transform(data.values.reshape(-1, 1))
