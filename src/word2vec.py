import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize

from data_preprocessor import DataPreprocessor


class Word2VecWrapper:

    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model = None

    def fit(self, data):
        sentences = self.data_preparing(data)
        self.model = Word2Vec(sentences, vector_size=500, window=5, min_count=5, epochs=10)

    def transform(self, data):
        sentences = self.data_preparing(data)
        embedded_sentences = np.array([])
        for sentence in tqdm(sentences):
            vector = [self.model.wv[word] for word in sentence
                      if word in list(self.model.wv.key_to_index.keys())]
            if vector:
                vector = np.mean(vector, axis=0)
                embedded_sentences = vector if embedded_sentences.size == 0 else np.vstack((embedded_sentences, vector))
        return embedded_sentences

    def data_preparing(self, data):
        return [self.data_preprocessor.run_pipeline(sent) for sent in sent_tokenize(' '.join(data), 'russian')]
