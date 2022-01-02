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

# Load model
filename = course_name + '.sav'
clf = pickle.load(open(filename, 'rb'))

# Data Preparation
def preprocess(sentence,language):
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    stop_words = stopwords.words(language)
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    if(language == 'en'):
        stemmer = SnowballStemmer("english")
        tokens = [stemmer.stem(word) for word in tokens]
    elif(language == 'tr'):
        stemmer = TurkishStemmer()
        tokens = [stemmer.stemWord(word) for word in tokens]
    return ' '.join(tokens)

X = [preprocess(text,language) for text in questions]

# Prediction
count_vect = CountVectorizer()
predictions = []
for sentence in questions:
    sentence_proc = preprocess(sentence,language)
    sentence_counts = count_vect.transform([sentence_proc])
    prediction = clf.predict(sentence_counts)
    predictions.append(prediction[0])
predictions = '|'.join(predictions)

# Return results
print(predictions)