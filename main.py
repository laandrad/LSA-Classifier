import os
import csv
import json
from sys import argv
from scipy import spatial
from tools import *
from numpy import isnan


def main():
    """this is the run file"""
    test_file = argv[1]
    train_folder = argv[2]
    method = argv[3]
    bigram = argv[4]
    output_path = argv[5]

    # read file paths from train folder
    query = sorted([os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.txt')])

    # load train features
    train_features = []
    for q in query:
        dtm1 = ExtractFeatures(method, bigram)
        train_features = train_features + dtm1.extract_features_from_file(q)

    # extract terms from each text document to create a vocabulary (keeping unique terms only)
    vocabulary = sorted(set(w[1] for w in train_features))
    documents = sorted(set(w[0] for w in train_features))
    print "{0} features produced.".format(str(len(vocabulary)))
    print "{0} documents read.".format(str(len(documents)))

    # write train vocabulary to a file
    vocabulary_file = output_path + '/' + 'train_vocabulary.txt'
    key = xrange(len(vocabulary))
    voc_dic = dict(zip(key, vocabulary))
    with open(vocabulary_file, 'w') as outfile:
        json.dump(voc_dic, outfile)

    # compute document-term-frequency matrix
    print "Computing frequency vectors for train set..."
    train_dtm = DTM(vocabulary, documents, train_features)
    train_dtm = train_dtm.compute_dtm()

    # write train features to a file
    train_file = output_path + '/' + 'train_set_dtm.txt'
    with open(train_file, 'w') as outfile:
        json.dump(train_dtm, outfile)

    print "Reading data from file: " + train_file
    with open(train_file, 'r') as infile:
        train_dtm = json.load(infile)
    print "Reading data from file: " + vocabulary_file
    with open(vocabulary_file, 'r') as infile:
        vocabulary = json.load(infile)
    vocabulary = [vocabulary[k] for k in vocabulary]
    print len(vocabulary)

    # initialize file writer and write the header with the document names
    # a 'cosine_values.csv' file will be written inside 'output_path' folder
    header = ['turn'] + train_dtm.keys()
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

    # for i in [0, 1]:
    for i in xrange(len(utterances)):
        dtm2 = ExtractFeatures(method, bigram)
        name = names[i] + "_" + str(i)
        test_features = dtm2.extract_features_from_text(utterances[i], name)

        test_dtm = DTM(vocabulary, name, test_features)
        test_dtm = test_dtm.compute_dtv()

        dv = []
        for k, v in train_dtm.iteritems():
            d = 1 - spatial.distance.cosine(test_dtm[name], train_dtm[k])
            if isnan(d):
                d = 0
            dv.append(d)

        line = [name] + dv

        with open(csm_file, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(line)
        print line
        print "Finished text " + str(i + 1) + " of " + str(len(utterances))


if __name__ == '__main__':
    main()
