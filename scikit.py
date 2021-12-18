import nltk
from nltk.corpus import stopwords
from snowballstemmer import TurkishStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Data Collection
X = [
    "Web sayfası, World Wide Web için hazırlanan ve web tarayıcısı kullanılarak görüntülenebilen dokümanlardır.",
    "Web sayfaları çoğunlukla HTML formatında kodlanır, CSS, betik, görsel ve diğer yardımcı kaynaklardan yararlanılarak son görünümüne sahip olur ve işlevsellik kazanır.",
    "Birden fazla web sayfasının bir araya gelmesi ile ortaya çıkan web sitesi ile karıştırılmamalıdır.",
    "Günlük konuşma dilinde internet sayfası terimi de çoğunlukla web sitesi anlamında kullanılmaktadır.",
    "Tipik bir web sayfası, diğer web sayfalarına hiper-bağlantıların bulunduğu bir hiper-metindir ancak farklı teknolojilerle de hazırlanmasında bir engel bulunmamaktadır.",
    "Web hazırlanan ve web tarayıcısı kullanılarak görüntülenebilen dokümanlardır.",
    "Web sayfaları çoğunlukla HTML formatında kodlanır, CSS, betik, görsel ve diğer olur ve işlevsellik kazanır.",
    "Birden sayfasının bir araya gelmesi ile ortaya sitesi ile karıştırılmamalıdır.",
    "Günlük internet sayfası terimi de çoğunlukla web sitesi anlamında kullanılmaktadır.",
    "Tipik bir web sayfası, diğer web sayfalarına hiper-bağlantıların bulunduğu bir hiper-metindir ancak farklı",
]
y = [0,1,2,3,4,0,1,2,3,4]
language = 'turkish'

# Data Preparation
def preprocess(sentence,language):
    sentence = sentence.lower()
    tokens = nltk.word_tokenize(sentence)
    stop_words = stopwords.words(language)
    tokens = [w for w in tokens if not w in stop_words and w.isalpha()]
    stemmer = TurkishStemmer()
    tokens = [stemmer.stemWord(word) for word in tokens]
    return ' '.join(tokens)

X = [preprocess(text,language) for text in X]

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

y_pred = clf.predict(X_test_counts)
print("Number of mislabeled points out of a total %d points : %d" % (X_test_counts.shape[0], (y_test != y_pred).sum()))


# Example Prediction
sentence = "CSS yönergeleri HTML kodları içerisinde verilebileceği gibi farklı dosyalarda da yer alabilir."
sentence_proc = preprocess(sentence,language)
sentence_counts = count_vect.transform([sentence_proc])
print(clf.predict(sentence_counts))


# Cross Selection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

tfidf = TfidfVectorizer()
cv_data = tfidf.fit_transform(X)

CV = 2
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, cv_data, y, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))

print(entries)