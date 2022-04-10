import sys
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess
import pickle

# Install NLTK dependencies
nltk.download('stopwords')
nltk.download('punkt')

# Read input data
course_name = sys.argv[1]
language = sys.argv[2]
amount = int(sys.argv[3])
questions = []
for i in range(4,amount+4):
    question = sys.argv[i]
    questions.append(question)

# Load model and vectorizer
filename = course_name + '.sav'
clf,vect = pickle.load(open(filename, 'rb'))

# Preprocess
questions = [' '.join(preprocess(text,language,stem=False)) for text in questions]

# Prediction
predictions = []
for sentence in questions:
    sentence_counts = vect.transform([sentence])
    prediction = clf.predict(sentence_counts)
    predictions.append(prediction[0])
predictions = '|'.join(predictions)

# Return results
print(predictions)