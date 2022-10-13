from asyncore import read
from collections import Counter
from os import listdir
from os.path import join
import re
from unidecode import unidecode
from matplotlib import pyplot as plt
from sklearn.manifold import MDS, TSNE

import numpy as np

def group_n(text, n):
    res = []
    for i in range(len(text)-n+1):
        res.append(text[i:i+n])
    return res

def terke(text, n):
    """
    Returns dict with counted tries of length n.
    """
    text = text.lower()
    text = unidecode(text)
    text = re.sub(r'[^A-Za-z ]+', '', text)
    text = re.sub(r'\n', '', text)
    res = dict(Counter(group_n(text, n)))
    return res


def read_data(n_terke):
    lds = {}
    for fn in listdir('languages'):
        if fn.lower().endswith('.txt'):
            with open(join('languages', fn), encoding='utf8') as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1:dict, d2:dict):
    skupni = set(d1.keys()).intersection(set(d2.keys()))
    produkt = sum(d1[key] * d2[key] for key in skupni)
    kolicnik = np.sqrt(sum(terka**2 for terka in d1.values())) * np.sqrt(sum(terka**2 for terka in d2.values()))
    if kolicnik == 0:
        return 1
    return 1 - produkt/kolicnik


def prepare_data_matrix(data_dict:dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """
    matrika = np.zeros((100, 100))
    terke_count = {}

    for dokument, terke_dict in data_dict.items():
        for terka, count in terke_dict.items():
            try:
                terke_count[terka] += 1
            except:
                terke_count[terka] = 1

    terke_count = {k: v for k, v in sorted(terke_count.items(), key=lambda item: item[1],reverse=True)} # sortira array
    top_terke = list(terke_count.keys())[0:100]
    docs = list(data_dict.keys())

    for iTerka, terka in enumerate(top_terke):
        for iDoc, doc in enumerate(docs):
            try:
                matrika[iDoc, iTerka] = data_dict[doc][terka]/sum(data_dict[doc].values())
            except:
                pass


    return matrika, [doc[0:2] for doc in docs]

def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    X = np.cov(np.transpose(X))
    vektor = np.random.rand(X.shape[1])

    for i in range(1000):
        produkt = np.dot(X, vektor)
        norma = np.linalg.norm(produkt)
        vektor = produkt / norma

    return vektor, (np.dot(X, vektor)/vektor)[0]



def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    vektor, value = power_iteration(X)
    projekcija = X - (vektor * np.transpose([project_to_eigenvectors(X, vektor)]))
    vektor2, value2 = power_iteration(projekcija)
    return np.array((vektor, -vektor2)), np.array((value, value2))


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    return np.dot(X - np.mean(X, axis=0), np.transpose(vecs))


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return np.sum(eigenvalues)/total_variance(X)


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    docs = read_data(3)
    X, languages = prepare_data_matrix(docs)
    vektorja, values = power_iteration_two_components(X)

    dims = [list(np.dot(vektor, np.transpose(X))) for vektor in vektorja]
    plt.scatter(dims[0], dims[1])
    for i in range(len(languages)):
        plt.annotate(languages[i], xy=(dims[0][i], dims[1][i]))
    plt.savefig('PCA-graph.png')
    plt.clf()
    


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    razdalja = np.zeros((100, 100))
    docs = read_data(3)
    X, languages = prepare_data_matrix(docs)
    for docIndex in range(len(docs.keys())):
        for terkaIndex in range(len(docs.keys())):
            if docIndex != terkaIndex:
                razdalja[docIndex][terkaIndex] = cosine_dist(docs[list(docs.keys())[docIndex]], docs[list(docs.keys())[terkaIndex]])

    mds = MDS(random_state=0, dissimilarity='precomputed')
    g1 = mds.fit_transform(razdalja)
    g2 = TSNE(n_components=2, perplexity=30.0, n_iter=1000, metric='precomputed').fit_transform(razdalja)
    plt.scatter(g1[:,0], g1[:,1])
    for i in range(len(languages)):
        plt.annotate(languages[i], xy=(g1[i,0], g1[i,1]))
    plt.savefig('MDS-graph.png')
    plt.clf()

    plt.scatter(g2[:,0], g2[:,1])
    for i in range(len(languages)):
        plt.annotate(languages[i], xy=(g2[i,0], g2[i,1]))
    plt.savefig('TSNE-graph.png')
    plt.clf()


if __name__ == "__main__":
   plot_MDS()
   plot_PCA()