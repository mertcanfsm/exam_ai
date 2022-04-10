import sys
import nltk
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Install NLTK dependencies
nltk.download('stopwords')
nltk.download('punkt')

# Read input data
course_name = sys.argv[1]
language = sys.argv[2]
y = sys.argv[3].split('|')
amount = int(sys.argv[4])
questions = []
for i in range(5,amount+5):
    question = sys.argv[i]
    questions.append(question)

# Preprocess
X = [' '.join(preprocess(text,language,stem=False)) for text in questions]

vect = TfidfVectorizer()
X_tfidf = vect.fit_transform(X)

clf = MultinomialNB().fit(X_tfidf, y)

# Store model and vectorizer
filename = course_name + '.sav'
pickle.dump((clf,vect), open(filename, 'wb'))