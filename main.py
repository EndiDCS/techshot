import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from gensim.parsing.preprocessing import remove_stopwords
# graphics import
import matplotlib.pyplot as plt
import seaborn as sns

 #Natural language tool kits
import nltk
from nltk import FreqDist, ngrams
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# download stopwords
nltk.download('stopwords')
import unidecode

from wordcloud import WordCloud,STOPWORDS

# string operations
import string
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.text import FreqDistVisualizer
from yellowbrick.text import TSNEVisualizer
from yellowbrick.contrib.classifier import DecisionViz
from yellowbrick.classifier import DiscriminationThreshold

from lime.lime_text import LimeTextExplainer

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import math

def create_subgraph(G, nodes):
    H = G.subgraph(nodes)
    return H

#FEATURES
def subdict(d, ks):
    new_dict = {}
    for k in ks:
        if k in d: #verifica se key existe no dicionario
            new_dict[k] = d[k]
    return new_dict


def degree_distribution(G, nodes=None):
    if nodes is not None:
        vk = dict(G.degree())
        vk = subdict(vk, nodes)
        vk = list(vk.values())  # we get only the degree values
    else:
        vk = dict(G.degree())
        vk = list(vk.values())  # we get only the degree values
    if len(vk) == 0:
        return [0], [0]

    maxk = np.max(vk)
    mink = np.min(min)
    kvalues = np.arange(0, maxk + 1)  # possible values of k
    Pk = np.zeros(maxk + 1)  # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk / sum(Pk)  # the sum of the elements of P(k) must to be equal to one
    return kvalues, Pk


def momment_of_degree_distribution(G, m, nodes=None):
    k, Pk = degree_distribution(G, nodes)
    if k[0] == 0 and len(k) == 1:
        return 0

    M = sum((k ** m) * Pk)
    return M

def sum_of_degrees(G,nodes = None):
    vk = dict(G.degree())
    vk = subdict(vk,nodes)
    vk = list(vk.values()) # we get only the degree values
    if len(vk) > 0:
        return sum(vk)
    else:
        return 0

def shannon_entropy(G,nodes=None):
    k,Pk = degree_distribution(G,nodes)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

#BUILD GRAPH
def plot_graph(G, title=None):
    # set figure size
    plt.figure(figsize=(10, 10))

    # define position of nodes in figure
    pos = nx.spring_layout(G, k=0.8)

    # draw nodes and edges
    nx.draw(G, pos=pos, with_labels=True)

    # get edge labels (if any)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # draw edge labels (if any)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # plot the title (if any)
    plt.title(title)

    plt.show()
    return

def get_relations(document):
    # in our case, relations are bigrams in sentences
    bigrams = []
    for sent in document:
        for i in range(len(sent)-1):
            # for every word and the next in the sentence
            pair = [sent[i], sent[i+1]]
            # only add unique bigrams
            if pair not in bigrams:
                bigrams.append(pair)
    return bigrams

def get_entities(document):
    # in our case, entities are all unique words
    unique_words = []
    for sent in document:
        for word in sent:
            if word not in unique_words:
                unique_words.append(word)
    return unique_words

def preprocess_document(document, sentence_spliter='.', word_spliter=' ', punct_mark=','):
    # lowercase all words and remove trailing whitespaces
    document = document.lower().strip()

    # remove unwanted punctuation marks
    for pm in punct_mark:
        document = document.replace(pm, '')

    # get list of sentences which are non-empty
    sentences = [sent for sent in document.split(sentence_spliter) if sent != '']

    # get list of sentences which are lists of words
    document = []
    for sent in sentences:
        words = sent.strip().split(word_spliter)
        document.append(words)

    return document

def build_graph(doc):
    # preprocess document for standardization
    pdoc = preprocess_document(doc)

    # get graph nodes
    nodes = get_entities(pdoc)

    # get graph edges
    edges = get_relations(pdoc)

    # create graph structure with NetworkX
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G



import pickle
################################## END IMPORTS ###########################################

sw = stopwords.words('english')


def clean_text(text):


    # remove
    text = re.sub('\[.*?\]', '', text)
    # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove tags
    text = re.sub('<.*?>+', '', text)
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove breaklines
    text = re.sub('\n', '', text)
    # remove numbers
    text = re.sub('\w*\d\w*', '', text)

    # remove accent
    text = unidecode.unidecode(text)

    # transform text into token
    text_token = nltk.word_tokenize(text)

    # remove stopwords
    words = [w for w in text_token if w not in sw]

    return ' '.join(words)

def lemmatize_sentence(text):


    # transform text into token
    text_token = nltk.word_tokenize(text)
    lemmatized_sentence = []
    for word in text_token:
        lemmatized_sentence.append(word)
    return " ".join(lemmatized_sentence)


def aplica_transformcacoes(df):
    df['review_lw'] = df['review'].str.lower()
    df['review_cl'] = df['review_lw'].apply(clean_text)
    nltk.download('omw-1.4')
    df['review_lm'] = df['review_cl'].apply(lemmatize_sentence)
    return df

def remove_stopwords2(text):

    vectorizer = CountVectorizer(min_df=0.2, max_df=0.8)
    docs = vectorizer.fit_transform(df['review_lm'].tolist())
    features = vectorizer.get_feature_names()
    stop_words_ = vectorizer.stop_words_

    # transform text into token
    text_token = nltk.word_tokenize(text)
    clean_text = []
    for word in text_token:
        if word not in stop_words_:
            clean_text.append(word)
    retorno = ' '.join(map(str,clean_text))
    return retorno


def get_top_n(word_dist, maximo=10):
    dicionario = dict(word_dist)
    value_key_pairs = ((value, key) for (key,value) in dicionario.items())
    sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)
    palavras = []
    for i in range(0,maximo):
        palavras.append(sorted_value_key_pairs[i][1])
    return palavras

def preprocess_dataset():
    nltk.download('omw-1.4')

    # Lê os dados
    path = '../IMDB.csv'
    df = pd.read_csv(path)
    # transformando dados para lowercase
    df = aplica_transformcacoes(df)
    # removmendo palavras de baixa frequencia
    df['review_sw'] = df['review_lm'].apply(remove_stopwords)
    df.to_csv('imdb_preprossed.csv')


def treina_modelo(df):
    X_train, X_test, y_train, y_test = train_test_split(df[['review_sw']], df['sentiment'], test_size=0.3,
                                                        random_state=42, stratify=df['sentiment'])

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print("-" * 20)
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # aplicando grafos
    df = pd.concat([X_train, y_train], axis=1)
    df.columns = ['review_sw', 'sentiment']
    n_words = [5000, 1000, 500, 100, 10]
    # positivas
    a = df[df['sentiment'] == 'positive']['review_sw'].str.lower().str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    positive_words = []
    for n in n_words:
        positive_words.append(get_top_n(word_dist, maximo=n))
    # negativas
    a = df[df['sentiment'] == 'negative']['review_sw'].str.lower().str.cat(sep=' ')
    words = nltk.tokenize.word_tokenize(a)
    word_dist = nltk.FreqDist(words)
    negative_words = []
    for k in n_words:
        negative_words.append(get_top_n(word_dist, maximo=k))

    key_positive_words = []
    key_negative_words = []
    for i in range(0, len(n_words)):
        a = set(positive_words[i])
        b = set(negative_words[i])
        key_positive_words.append(a - b)
        key_negative_words.append(b - a)

    new_x_train = []
    for index, row in X_train.iterrows():
        G = build_graph(row.values[0])
        features = []
        features.append(round(shannon_entropy(G), 2))
        features.append(round(momment_of_degree_distribution(G, 1), 2))
        # positive features
        for positive in key_positive_words:
            features.append(round(shannon_entropy(G, positive), 2))
            features.append(round(sum_of_degrees(G, positive), 2))
            features.append(round(momment_of_degree_distribution(G, 1, positive), 2))
        # negative features
        for negative in key_negative_words:
            features.append(round(shannon_entropy(G, negative), 2))
            features.append(round(sum_of_degrees(G, negative), 2))
            features.append(round(momment_of_degree_distribution(G, 1, negative), 2))
        new_x_train.append(features)

    new_x_test = []
    for index, row in X_test.iterrows():
        G = build_graph(row.values[0])
        features = []
        features.append(round(shannon_entropy(G), 2))
        features.append(round(momment_of_degree_distribution(G, 1), 2))
        # positive features
        for positive in key_positive_words:
            features.append(round(shannon_entropy(G, positive), 2))
            features.append(round(sum_of_degrees(G, positive), 2))
            features.append(round(momment_of_degree_distribution(G, 1, positive), 2))
        # negative features
        for negative in key_negative_words:
            features.append(round(shannon_entropy(G, negative), 2))
            features.append(round(sum_of_degrees(G, negative), 2))
            features.append(round(momment_of_degree_distribution(G, 1, negative), 2))
        new_x_test.append(features)

    fig = plt.figure(figsize=(15, 10))
    clf = MultinomialNB().fit(new_x_train, y_train)
    visualizer = ConfusionMatrix(clf, percent=True)
    visualizer.score(new_x_test, y_test)
    visualizer.show()

    # Store data (serialize)
    with open('model.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #salva palavras positivas e negativas
    with open('positive.pickle', 'wb') as handle:
        pickle.dump(key_positive_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('negative.pickle', 'wb') as handle:
        pickle.dump(key_negative_words, handle, protocol=pickle.HIGHEST_PROTOCOL)


def aplica_transformacoes(entrada):
    nltk.download('omw-1.4')
    nltk.download('punkt')

    with open("positive.pickle", "rb") as input_file:
        key_positive_words = pickle.load(input_file)
    with open("negative.pickle", "rb") as input_file:
        key_negative_words = pickle.load(input_file)
    new_df = pd.DataFrame(entrada)
    new_df.columns = ['review']
    df = aplica_transformcacoes(new_df)
    # removmendo palavras de baixa frequencia
    df['review_sw'] = df['review_lm'].apply(remove_stopwords)
    new_x_test = []
    for index, row in df.iterrows():
        G = build_graph(row.values[0])
        features = []
        features.append(round(shannon_entropy(G), 2))
        features.append(round(momment_of_degree_distribution(G, 1), 2))
        # positive features
        for positive in key_positive_words:
            features.append(round(shannon_entropy(G, positive), 2))
            features.append(round(sum_of_degrees(G, positive), 2))
            features.append(round(momment_of_degree_distribution(G, 1, positive), 2))
        # negative features
        for negative in key_negative_words:
            features.append(round(shannon_entropy(G, negative), 2))
            features.append(round(sum_of_degrees(G, negative), 2))
            features.append(round(momment_of_degree_distribution(G, 1, negative), 2))
        new_x_test.append(features)

    return new_x_test


if __name__ == '__main__':
    preprocess = False
    treinar = False
    if preprocess:
        preprocess_dataset()

    if treinar:
        df = pd.read_csv('imdb_preprossed.csv')
        treina_modelo(df)

    with open("model.pickle", "rb") as input_file:
        model = pickle.load(input_file)
    entrada = ['this is a review of a film that is very good i loved the movie']
    new_x_test = aplica_transformacoes(entrada)
    print(model.predict(new_x_test))


