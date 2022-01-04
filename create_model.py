import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from snowballstemmer import TurkishStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Read input data
course_name = sys.argv[0]
language = sys.argv[1]
y = sys.argv[2].split('|')
amount = sys.argv[3]
questions = []
for i in range(4,amount+4):
    question = sys.argv[i]
    questions.append(question)

# Data Preparation
def preprocess(sentence,language):
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    stop_words = stopwords.words(language)
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    if(language == 'turkish'):
        stemmer = TurkishStemmer()
        tokens = [stemmer.stemWord(word) for word in tokens]
    else:
        stemmer = SnowballStemmer(language)
        tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

X = [preprocess(text,language) for text in questions]

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Store model and vectorizer
filename = course_name + '.sav'
pickle.dump((clf,count_vect), open(filename, 'wb'))