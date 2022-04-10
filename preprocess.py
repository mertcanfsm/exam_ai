import nltk
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer
from nltk.stem.snowball import SnowballStemmer

def preprocess(sentence,language,stem=True):
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    stop_words = stopwords.words(language)
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    if stem:
        if(language == 'turkish'):
            stemmer = TurkishStemmer()
            tokens = [stemmer.stemWord(word) for word in tokens]
        else:
            stemmer = SnowballStemmer(language)
            tokens = [stemmer.stem(word) for word in tokens]
    return tokens