import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from model_wrappers import OHEWrapper
from model_wrappers import Word2VecWrapper
from utils import text_preproc, csv_reader


class TextModel:
    def __init__(self, dir_path, filename, vectorizer):
        self.dir_path = dir_path
        self.filename = filename.split('.')[0]
        self.vectorizer = vectorizer

        file_path = os.path.join(dir_path, filename)
        self.data = csv_reader(file_path)

        self.model = None
        self.svd_model = TruncatedSVD(n_components=300)
        self.knn_model = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree')
        self.run_model_pipeline()

    def run_model_pipeline(self):
        self.init_model()
        print(f'{self.vectorizer} Model initialized')
        reduced_embedding = self.fit_transform()
        print(f'{self.vectorizer} Model trained')
        self.train_knn(reduced_embedding)
        print('KNN trained')

    def init_model(self):
        if self.vectorizer == 'OHE':
            self.model = OHEWrapper()
        elif self.vectorizer == 'BOW':
            self.model = CountVectorizer()
        elif self.vectorizer == 'TFIDF':
            self.model = TfidfVectorizer()
        elif self.vectorizer == 'W2V':
            self.model = Word2VecWrapper()

    def fit_transform(self):
        self.model.fit(self.data.context_0)
        embedding = self.model.transform(self.data.context_0)
        reduced_embedding = self.reduce_dim(embedding)
        return reduced_embedding

    def reduce_dim(self, embedding):
        reduced_embedding = self.svd_model.fit_transform(embedding)
        return reduced_embedding

    def train_knn(self, reduced_embedding):
        self.knn_model.fit(reduced_embedding, self.data.reply)

    def predict_answer(self, question):
        embedding = self.model.transform(question)
        reduced_embedding = self.svd_model.transform(embedding)
        answer = self.knn_model.predict(reduced_embedding)
        return answer, reduced_embedding

    def check_test(self, dir_path, filename):
        print('Start to predict')

        start_time = time()
        test_data = csv_reader(os.path.join(dir_path, filename))
        questions = test_data['context_0']
        model_answers, _ = self.predict_answer(questions)

        print(f'For {len(questions)} questions received {len(model_answers)} answers')
        print(f'Spent {time() - start_time} seconds')

        score = 0
        for question, answer in tqdm(zip(questions, model_answers)):
            good_answers = test_data[test_data['context_0'] == question]['reply'].to_list()
            if answer in good_answers:
                score += 1
        precision = score / len(questions)

        print(f'Score is {precision * 100}%')
        return precision

    def embedding_analyzer(self, dir_path, filename):
        print('Start to predict')

        start_time = time()
        test_data = csv_reader(os.path.join(dir_path, filename))
        questions = test_data['context_0']
        _, reduced_embedding = self.predict_answer(questions)
        print(f'Spent {time() - start_time} seconds')

        tsne_model = TSNE(n_components=2, random_state=42, n_iter=250, metric='cosine')
        # learning_rate='auto', init='pca', square_distances=True
        tsne_embedding = tsne_model.fit_transform(reduced_embedding)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], alpha=.1)
        plt.title('Scatter plot of games using t-SNE')
        plt.savefig(f'../images/tsne_embedding_{self.vectorizer}.png')
        plt.show()


if __name__ == '__main__':

    filepath = f'../data'
    source_filename = f'good_clean.tsv'
    # source_filename = 'good_test_dataset1.csv'
    test_filename = 'good_test_dataset.tsv'
    # test_filename = 'good_test_dataset1.csv'

    text_model = TextModel(filepath, source_filename, vectorizer='OHE')
    # text_model.check_test(filepath, test_filename)
    text_model.embedding_analyzer(filepath, test_filename)


