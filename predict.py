import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from snowballstemmer import TurkishStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Read input data
course_name = sys.argv[0]
language = sys.argv[1]
amount = sys.argv[2]
questions = []
for i in range(3,amount+3):
    question = sys.argv[i]
    questions.append(question)

# Load model and vectorizer
filename = course_name + '.sav'
clf,count_vect = pickle.load(open(filename, 'rb'))

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

questions = [preprocess(text,language) for text in questions]

# Prediction
predictions = []
for sentence in questions:
    sentence_counts = count_vect.transform([sentence])
    prediction = clf.predict(sentence_counts)
    predictions.append(prediction[0])
predictions = '|'.join(predictions)

# Return results
print(predictions)