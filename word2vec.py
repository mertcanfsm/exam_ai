# Model: https://github.com/eyaler/word2vec-slim/blob/master/GoogleNews-vectors-negative300-SLIM.bin.gz

from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

file_name = 'GoogleNews-vectors-negative300-SLIM.bin'

question = "What is Anchor tag and how can you open an URL into a new tab when clicked?"

learning_outcomes = [
'Be able to use the HTML programming language',
'Resolves written HTML codes',
'Runs the page he she has designed using HTML codes',
'Be able to use the Design Programs',
'Uses Microsoft Expression Web 4 programme',
'Designs site and page via Microsoft Expression Web 4 programme',
'Uses the program Web Page Maker',
'Designs site and page via Web Page Maker programme',
'Publishes the site he she designed',
'Uses Domain Name and services',
'Be able to make changes on the Site',
'Updates on the site when needed'
]

model = KeyedVectors.load_word2vec_format(file_name, binary=True)
keys = model.index_to_key

def preprocess(sentence):
    remove = []
    tokens = remove_stopwords(strip_punctuation(sentence)).lower().split()
    for word in tokens:
        if word not in keys:
            remove.append(word)
    for word in remove:
        tokens.remove(word)
    return tokens

question = preprocess(question)

i = 0
most_similar = 0
max_similarity = 0
for outcome in learning_outcomes:
    similarity = model.n_similarity(question, preprocess(outcome))
    if similarity > max_similarity:
        most_similar = i
        max_similarity = similarity
    i = i + 1

print(learning_outcomes[most_similar])