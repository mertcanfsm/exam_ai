from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

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
X = [' '.join(preprocess(text,language,stem=False)) for text in X]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# Vectorize
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Types of classifiers
models = [
    RandomForestClassifier(random_state=0),
    LinearSVC(random_state=0),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# Compare model accuracies
results = []
for model in models:
    clf = model.fit(X_train_tfidf, y_train)
    accuracy = clf.score(X_test_tfidf, y_test)
    model_name = model.__class__.__name__
    results.append(model_name + ": " + str(accuracy))

# Print results
print('\n'.join(results))