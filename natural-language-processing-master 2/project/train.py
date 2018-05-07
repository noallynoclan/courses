import os
import sys
import project.utils as utils
import numpy as np
import pandas as pd
import pickle
from common.download_utils import download_project_resources
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

download_project_resources()


def tfidf_features(x_train, x_test, vectorizer_path):
    """Performs TF-IDF transformation and dumps the model."""
    # Train a vectorizer on X_train data.
    # Transform X_train and X_test data.
    # Pickle the trained vectorizer to 'vectorizer_path'
    # Don't forget to open the file in writing bytes mode.
    vect = TfidfVectorizer()
    vect.fit(x_train)
    x_train = vect.transform(x_train)
    x_test = vect.transform(x_test)
    with open(vectorizer_path, "wb") as file:
        pickle.dump(vect, file)
    return x_train, x_test


print('1. Data preparation ...')
sample_size = 200000
dialogue_df = pd.read_csv('data/dialogues.tsv', sep='\t').sample(sample_size, random_state=0)
stackoverflow_df = pd.read_csv('data/tagged_posts.tsv', sep='\t').sample(sample_size, random_state=0)
dialogue_df['text'] = dialogue_df.text.apply(utils.text_prepare)
stackoverflow_df['title'] = stackoverflow_df.title.apply(utils.text_prepare)
X = np.concatenate([dialogue_df['text'].values, stackoverflow_df['title'].values])
y = ['dialogue'] * dialogue_df.shape[0] + ['stackoverflow'] * stackoverflow_df.shape[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.1)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))
X_train_tfidf, X_test_tfidf = tfidf_features(X_train, X_test, "tfidf_vectorizer.pkl")

print('2. Intent recognition ...')
intent_recognizer = LogisticRegression(penalty="l2", C=10, random_state=0)
intent_recognizer.fit(X_train_tfidf, y_train)
y_test_pred = intent_recognizer.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))
pickle.dump(intent_recognizer, open(utils.RESOURCE_PATH['INTENT_RECOGNIZER'], 'wb'))

print('3. Programming language classification ...')
X = stackoverflow_df['title'].values
y = stackoverflow_df['tag'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('Train size = {}, test size = {}'.format(len(X_train), len(X_test)))
vectorizer = pickle.load(open(utils.RESOURCE_PATH['TFIDF_VECTORIZER'], 'rb'))
X_train_tfidf, X_test_tfidf = vectorizer.transform(X_train), vectorizer.transform(X_test)
tag_classifier = OneVsRestClassifier(LogisticRegression(penalty="l2", C=5, random_state=0))
tag_classifier.fit(X_train_tfidf, y_train)
y_test_pred = tag_classifier.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print('Test accuracy = {}'.format(test_accuracy))
pickle.dump(tag_classifier, open(utils.RESOURCE_PATH['TAG_CLASSIFIER'], 'wb'))

print('4. Ranking questions with embeddings ...')
_, _, embeddings, _, embeddings_dim = utils.load_embeddings()
posts_df = pd.read_csv('data/tagged_posts.tsv', sep='\t')
counts_by_tag = posts_df.groupby("tag").count().max(axis=1)
os.makedirs(utils.RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], exist_ok=True)
for tag, count in counts_by_tag.items():
    tag_posts = posts_df[posts_df['tag'] == tag]
    tag_post_ids = posts_df[posts_df['tag'] == tag].post_id
    tag_vectors = np.zeros((count, embeddings_dim), dtype=np.float32)
    for i, title in enumerate(tag_posts['title']):
        tag_vectors[i, :] = utils.question_to_vec(title, embeddings, embeddings_dim)
    # Dump post ids and vectors to a file.
    filename = os.path.join(utils.RESOURCE_PATH['THREAD_EMBEDDINGS_FOLDER'], os.path.normpath('%s.pkl' % tag))
    pickle.dump((tag_post_ids, tag_vectors), open(filename, 'wb'))
