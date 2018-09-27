import email
import email.policy
import os
import re
from email.parser import BytesParser

import nltk
import numpy as np
from collections import Counter

import urlextract
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ApacheSpamAssassin:

    def loadEmails(self, dirs):
        emails = []
        for dir in dirs:
            filenames = [name for name in sorted(os.listdir(dir)) if len(name) > 20]
            for filename in filenames:
                # msg = email.message_from_file(os.path.join(dir, filename))
                fullname = os.path.join(dir, filename)
                with open(fullname, "rb") as fe:
                    msg = BytesParser(policy=email.policy.default).parse(fe)
                    emails.append(self.email_to_text(msg))
        return emails

    def prepared_data(self):
        X_train, X_test, y_train, y_test = self.data()
        preprocess_pipeline = Pipeline([
            ("email_to_wordcount", EmailToWordCounterTransformer()),
            ("wordcount_to_vector", WordCounterToVectorTransformer()),
        ])
        X_train_transformed = preprocess_pipeline.fit_transform(X_train)
        X_test_transformed = preprocess_pipeline.fit_transform(X_test)
        return X_train_transformed, X_test_transformed, y_train, y_test

    def data(self):
        # ham_dirs = ['../data/ApacheSpamAssassin/easy_ham', '../data/ApacheSpamAssassin/easy_ham_2', '../data/ApacheSpamAssassin/hard_ham']
        # ham_dirs = ['../data/ApacheSpamAssassin/easy_ham']
        ham_dirs = ['../data/ApacheSpamAssassin/hard_ham']
        ham_emails = self.loadEmails(ham_dirs)
        # print(len(ham_emails))
        spam_dirs = ['../data/ApacheSpamAssassin/spam', '../data/ApacheSpamAssassin/spam_2']
        # spam_dirs = ['../data/ApacheSpamAssassin/spam']
        spam_emails = self.loadEmails(spam_dirs)
        # print(len(spam_emails))
        from sklearn.model_selection import train_test_split
        X = np.array(ham_emails + spam_emails)
        y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def email_to_text(self, email):
        from bs4 import BeautifulSoup
        html = None
        for part in email.walk():
            ctype = part.get_content_type()
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except:  # in case of encoding issues
                content = str(part.get_payload())
            if ctype == "text/plain":
                return content
            else:
                html = content
        if html:
            return BeautifulSoup(html, 'html.parser').get_text()




class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        url_extractor = urlextract.URLExtract()
        stemmer = nltk.PorterStemmer()
        X_transformed = []
        for email in X:
            text = email or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                # urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.most_common_ = most_common
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
