import csv
import sys

from scipy import spatial

from myTMtools import *

from itertools import chain


def main():
    benchmark_path = sys.argv[1]
    transcription_path = sys.argv[2]
    output_path = sys.argv[3]
    title = sys.argv[4]
    method = sys.argv[5]
    bigrams = sys.argv[6]

    # get a path to the text files in 'input_path' folder
    query = sorted(get_imlist(benchmark_path))

    # extract subject name for each text document
    subject = [query[i][-13:-4] for i in range(len(query))]

    # read the text documents and convert words into tokens,
    # then bind all (tokens, subject) pairs into a list
    terms_by_benchmark = []
    for i in range(len(subject)):
        print 'Preparing text vector for: ', subject[i]
        terms = category_tokens(query[i], subject[i], method, bigrams)
        terms_by_benchmark = terms_by_benchmark + terms

    # initialize cosine similarity matrix file writer and write the header with the subject names
    # a 'title_cosine.csv' file will be written inside 'output_path' folder
    header = [['time'], ['subject'], subject]
    header = list(chain.from_iterable(header))
    csm_file = output_path + '/' + str(title) + '_cosine.csv'
    with open(csm_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)

    # load transcript file
    transcription = io.open(transcription_path, 'r', encoding='windows-1252')

    for line in transcription:
        tokens = nltk.word_tokenize(line)
        if not 0 >= len(tokens):

            # create a term list for line
            name = str(tokens[1])
            print 'Preparing text vector for line at time: ', name

            s = ' '
            text = s.join(tokens[6:len(tokens)])
            text = prepare_text(str(text.encode("utf-8")), method, bigrams)

            if not 0 >= len(text):
                category = []
                for i in range(len(text)):
                    category.append(name)

                terms = zip(category, text)

                # bind line terms with benchmark terms
                terms_by_category = terms_by_benchmark + terms
                # print terms_by_category

                # extract terms from each text document to create a vocabulary (keeping unique terms only)
                vocabulary = sorted(set(w[1] for w in terms_by_category))
                documents = sorted(set(w[0] for w in terms_by_category))

                # 1. extract term frequencies from each text document - a.k.a. frequency vectors
                # 2. bind all frequency vectors into a dtm matrix
                dtm = []
                for d in documents:
                    a = [word for category, word in terms_by_category if category == d]
                    dtv = [a.count(word) for word in vocabulary]  # vector of frequencies per vocabulary term
                    dtm.append(dtv)
                # print dtm

                dv = []
                for j in range(1, len(documents)):
                    d = 1 - spatial.distance.cosine(dtm[0], dtm[j])
                    dv.append(d)

                line = [[name], [tokens[4]], dv]
                line = list(chain.from_iterable(line))
                with open(csm_file, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    writer.writerow(line)
                print line


if __name__ == '__main__':
    main()
