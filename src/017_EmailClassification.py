from collections import Counter

import nltk
import numpy as np
import re
import urlextract
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ApacheSpamAssassin import ApacheSpamAssassin

# X_train, X_test, y_train, y_test = ApacheSpamAssassin().data()

# url_extractor = urlextract.URLExtract()
# stemmer = nltk.PorterStemmer()
#
# class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
#                  replace_urls=True, replace_numbers=True, stemming=True):
#         self.strip_headers = strip_headers
#         self.lower_case = lower_case
#         self.remove_punctuation = remove_punctuation
#         self.replace_urls = replace_urls
#         self.replace_numbers = replace_numbers
#         self.stemming = stemming
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X, y=None):
#         X_transformed = []
#         for email in X:
#             text = email or ""
#             if self.lower_case:
#                 text = text.lower()
#             if self.replace_urls and url_extractor is not None:
#                 urls = list(set(url_extractor.find_urls(text)))
#                 # urls.sort(key=lambda url: len(url), reverse=True)
#                 for url in urls:
#                     text = text.replace(url, " URL ")
#             if self.replace_numbers:
#                 text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
#             if self.remove_punctuation:
#                 text = re.sub(r'\W+', ' ', text, flags=re.M)
#             word_counts = Counter(text.split())
#             if self.stemming and stemmer is not None:
#                 stemmed_word_counts = Counter()
#                 for word, count in word_counts.items():
#                     stemmed_word = stemmer.stem(word)
#                     stemmed_word_counts[stemmed_word] += count
#                 word_counts = stemmed_word_counts
#             X_transformed.append(word_counts)
#         return np.array(X_transformed)
#
# class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, vocabulary_size=1000):
#         self.vocabulary_size = vocabulary_size
#     def fit(self, X, y=None):
#         total_count = Counter()
#         for word_count in X:
#             for word, count in word_count.items():
#                 total_count[word] += min(count, 10)
#         most_common = total_count.most_common()[:self.vocabulary_size]
#         self.most_common_ = most_common
#         self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
#         return self
#     def transform(self, X, y=None):
#         rows = []
#         cols = []
#         data = []
#         for row, word_count in enumerate(X):
#             for word, count in word_count.items():
#                 rows.append(row)
#                 cols.append(self.vocabulary_.get(word, 0))
#                 data.append(count)
#         return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


# preprocess_pipeline = Pipeline([
#     ("email_to_wordcount", EmailToWordCounterTransformer()),
#     ("wordcount_to_vector", WordCounterToVectorTransformer()),
# ])
#
# X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from ApacheSpamAssassin import ApacheSpamAssassin
X_train, X_test, y_train, y_test = ApacheSpamAssassin().prepared_data()

log_clf = LogisticRegression(random_state=42)
score = cross_val_score(log_clf, X_train, y_train, cv=3, verbose=3)
print(score)

