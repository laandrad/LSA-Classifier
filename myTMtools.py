from types import StringType

import io
import nltk
import os


def get_imlist(path):
    # This function extracts a path toward each txt file inside 'path' folder
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]


def category_tokens(path, name, method, bigrams):
    # This function does two things:
    # 1. read a text document from 'path' and convert words into token format 
    # 2. Use 'name' argument to label to the extracted tokens
    with io.open(path, 'r', encoding='windows-1252') as doc:
        raw = doc.read()

    tokens = prepare_text(str(raw), method, bigrams)

    category = []
    for i in range(len(tokens)):
        category.append(name)

    return zip(category, tokens)


def prepare_text(raw_text, method=1, bigrams=0):
    # type: (raw_text, method, bigrams) -> features
    # This function does four things to retrieve only names and nouns in 'raw_text':
    # 1. retrieve a list of words from a 'raw_text' file
    # 2. tokenize words in list using NLTK 
    # 3. create part-of-speech (POS) tags per token
    # 4. stem tokens using the Porter stemmer filter
    # 5. method 1 uses nouns only, method 2 uses nouns and adverbs
    # 6. Bigrams, boolean variable, that would additionally include bigrams
    """

    :rtype: features
    """
    assert isinstance(raw_text, StringType), "text is not a string: %r" % raw_text
    raw = raw_text

    porter = nltk.PorterStemmer()

    nouns = ['NN', 'NNP', 'NNS']
    advs = ['JJ', 'VBN']
    if method == '1':
        pos_tags = nouns
    else:
        pos_tags = nouns + advs

    sentences = raw.split('.')

    document = []
    for s in sentences:
        tokens = nltk.word_tokenize(s)
        words = [w.lower() for w in tokens if w.isalpha()]
        # print 'creating POS tags'
        tags = nltk.pos_tag(words)
        names = [w.lower() for w, t in tags if t in pos_tags]
        names = [porter.stem(t) for t in names]

        if bigrams == '1':
            relations = list(nltk.bigrams(names))
        else:
            relations = []

        document = document + names + relations

    return document
