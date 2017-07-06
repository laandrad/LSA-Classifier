import os
import sys
import csv
from scipy import spatial
from tools import *


def main():
    """this is the run file"""
    test_file = sys.argv[1]
    train_folder = sys.argv[2]
    method = sys.argv[3]
    bigram = sys.argv[4]
    output_path = sys.argv[5]

    # read file paths from train folder
    query = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.txt')])

    # load train features
    train_features = []
    for q in query:
        dtm1 = ExtractFeatures(method, bigram)
        train_features = train_features + dtm1.extract_features_from_file(q)

    # initialize file writer and write the header with the document names
    # a 'cosine_values.csv' file will be written inside 'output_path' folder
    docs = sorted(set(w[0] for w in train_features))
    header = ['turn'] + docs
    csm_file = output_path + '/' + 'cosine_values.csv'
    with open(csm_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)

    # load test text file
    with io.open(test_file, "r", encoding="utf-8", errors='ignore') as doc:
        raw = doc.read()

    raw = raw.split(":")

    names = [raw[i] for i in range(0, len(raw), 2)]
    utterances = [raw[i] for i in range(1, len(raw), 2)]

    for i in xrange(len(utterances)):
        dtm2 = ExtractFeatures(method, bigram)
        name = names[i] + "_" + str(i)
        test_features = dtm2.extract_features_from_text(utterances[i], name)

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

        dv = []
        for j in range(0, len(documents)-1):
            # print documents[-1], documents[j]
            d = 1 - spatial.distance.cosine(dtm[-1], dtm[j])
            dv.append(d)

        line = [documents[-1]] + dv

        with open(csm_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(line)
        print line
        print "Finished text " + str(i + 1) + " of " + str(len(utterances))

if __name__ == '__main__':
    main()
