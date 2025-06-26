from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

newgroups_train = fetch_20newsgroups(subset='train')
newgroups_test = fetch_20newsgroups(subset='test')

pipeline = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=3000))

pipeline.fit(newgroups_train.data, newgroups_train.target)

y_pred = pipeline.predict(newgroups_test.data)

print("Accuracy: %.2f" % accuracy_score( newgroups_test.target, y_pred ))