import os
import sys
from tools import *


def main():
    """this is the run file"""
    test_file = sys.argv[1]
    train_folder = sys.argv[2]
    method = sys.argv[3]
    bigram = sys.argv[4]

    query = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.txt')])

    train_features = []
    for q in query:
        with io.open(q, "r", encoding="utf-8", errors='ignore') as doc:
            raw = doc.read()
        dtm1 = ExtractFeatures(raw, method, bigram)
        train_features = train_features + dtm1.load_text()

    with io.open(test_file, "r", encoding="utf-8", errors='ignore') as doc:
        raw = doc.read()

    utterances = raw.split(":")

    for u in utterances:
        dtm2 = ExtractFeatures(u, method, bigram)
        test_features = dtm2.load_text()

        features = test_features + train_features

        # extract terms from each text document to create a vocabulary (keeping unique terms only)
        vocabulary = sorted(set(w[1] for w in features))
        documents = sorted(set(w[0] for w in features))

        # 1. extract term frequencies from each text document - a.k.a. frequency vectors
        # 2. bind all frequency vectors into a dtm matrix
        dtm = []
        for d in documents:
            a = [word for category, word in features if category == d]
            dtv = [a.count(word) for word in vocabulary]  # vector of frequencies per vocabulary term
            dtm.append(dtv)
        # print dtm

        dv = []
        for j in range(1, len(documents)):
            d = 1 - spatial.distance.cosine(dtm[0], dtm[j])
            dv.append(d)

        print dv

if __name__ == '__main__':
    main()
