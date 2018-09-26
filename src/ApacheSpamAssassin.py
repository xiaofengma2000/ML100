import email
import email.policy
import os
from email.parser import BytesParser
import numpy as np
from collections import Counter


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

    def data(self):
        # ham_dirs = ['../data/ApacheSpamAssassin/easy_ham', '../data/ApacheSpamAssassin/easy_ham_2', '../data/ApacheSpamAssassin/hard_ham']
        ham_dirs = ['../data/ApacheSpamAssassin/easy_ham']
        ham_emails = self.loadEmails(ham_dirs)
        # print(len(ham_emails))
        # spam_dirs = ['../data/ApacheSpamAssassin/spam', '../data/ApacheSpamAssassin/spam_2']
        spam_dirs = ['../data/ApacheSpamAssassin/spam']
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
