import email
import email.policy
import os
from email.parser import BytesParser

def loadEmails(dirs):
    emails = []
    for dir in dirs:
        filenames = [name for name in sorted(os.listdir(dir)) if len(name) > 20]
        for filename in filenames:
            # msg = email.message_from_file(os.path.join(dir, filename))
            fullname = os.path.join(dir, filename)
            with open(fullname, "rb") as fe:
                msg = BytesParser().parse(fe)
                emails.append(msg)
    return emails

ham_dirs = ['../data/ApacheSpamAssassin/easy_ham', '../data/ApacheSpamAssassin/easy_ham_2', '../data/ApacheSpamAssassin/hard_ham']
ham_emails = loadEmails(ham_dirs)
print(len(ham_emails))

spam_dirs = ['../data/ApacheSpamAssassin/spam', '../data/ApacheSpamAssassin/spam_2']
spam_emails = loadEmails(spam_dirs)
print(len(spam_emails))

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


