import nltk
import pickle
import re
import numpy as np
from urllib.request import urlopen

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))
    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])
    return text.strip()


def load_embeddings(n_words=10000):
    path = 'https://raw.githubusercontent.com/noallynoclan/data/master/'
    filename = path + 'glove.6B.50d.10k.txt'
    n_dims, n_tokens = 50, 2
    word2vec = {'<pad>': np.zeros(n_dims, dtype=np.float32),
                '<unk>': np.random.randn(n_dims).astype(np.float32)}
    word2id = {'<pad>': 0, '<unk>': 1}
    for n, line in enumerate(urlopen(filename)):
        if n >= n_words:
            break
        word, *embedding = line.decode('utf8').split()
        word2vec[word] = np.asarray(embedding, dtype=np.float32)
        word2id[word] = n_tokens + n
    id2word = dict((idx, word) for word, idx in word2id.items())
    embedding_matrix = np.zeros((n_tokens + n_words, n_dims), dtype=np.float32)
    for word, n in word2id.items():
        embedding_matrix[n] = word2vec[word]
    print('Word embeddings:', len(word2vec))
    print('Embedding matrix shape:', embedding_matrix.shape)
    return word2id, id2word, word2vec, embedding_matrix, n_dims


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    # Hint: you have already implemented exactly this function in the 3rd assignment.
    if question == "":
        return np.zeros(dim)
    t = np.array([embeddings[i] for i in question.split() if i in embeddings])
    if len(t) == 0:
        return np.zeros(dim)
    return t.mean(axis=0)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
