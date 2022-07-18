from turtle import shape
import pandas as pd
import re
import numpy as np
import classifiers as classifier
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


"""

Jia Wu, Khadija Jallow, & Tadius Frank 
May 3, 2022

Preprocessing sentiment data and calling classifiers to evaluate and 
graph classifers 

"""

fstem = True #boolean for word stems
train= 0.7
stop_words = stopwords.words("english")  # gets stopwords
stemmer = SnowballStemmer("english")  # gets stems

# tokenize and remove words with no sentimemtns 
def remove_stopwords(text):
    text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ',str(text).lower()).strip() #removes stopwords: Stop words are a set of commonly used words in a language that aren't useful in NLP
    if fstem:
        return " ".join([stemmer.stem(token) for token in text.split() if token not in stop_words])
    else:
        return " ".join([token for token in text.split() if token not in stop_words])

# tokenizes words and calls classifiers to evalute 
def preprocessData():
    
    tweets = pd.read_csv("twittersentiment.csv", encoding="ISO-8859-1", names = ["target", "ids", "date", "flag", "user", "text"], ) 

    # check if data is skewed 
    # skewvalue = tweets.skew(axis=0)
    # print(tweets['target'].count())
    # print(skewvalue)

    labels = {0: 0, 4: 1} #  {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"} relabel to 0 and 1
    tweets.target = tweets.target.apply(lambda x: labels[x]) #labels the data as pos, neg
    tweets.text = tweets.text.apply(remove_stopwords) # removes stop words and all "neutral" classes 
    vectorizer = TfidfVectorizer() # fcn to get count of words
    frequency = vectorizer.fit_transform(tweets.text) #frequency of each features
    features = vectorizer.get_feature_names_out()
    index = np.random.random(tweets.shape[0])
    # print(len(features))
    # print(tweets['target'].count())

    # X is features and Y is label, get all of x and all of y for k fold validation   
    x = frequency
    y = tweets.target 
    print(shape(x[0]))
    print(shape(y))

    X_train = frequency[index <= train, :]
    Y_train = tweets.target[index<= train]
    
    # testing the two algorithsm paramters
    # classifier.testLogisticReg(X_train, Y_train, x, y)
    # classifier.testDT(X_train, Y_train, x, y)

    # comparing the two algorithms's by best performer 
    clf = LogisticRegression(C=1, penalty="l1", solver='liblinear')
    # clf.fit(X_train,Y_train)
    # classifier.kfold_all(clf, x, y, "L1 with lambda = 1")

    # clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=200, min_samples_leaf=60)
    # classifier.kfold_all(clf, x, y, "Entropy DT with depth of 200 & min samples of 60")

    # clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=5, min_samples_leaf=60)
    # clf = clf.fit(X_train, Y_train)
    # classifier.kfold_score(clf, x, y, "Entropy DT with depth of 5 & min samples of 60")
    # classifier.plotDT(clf, features, ['NEGATIVE', 'POSITIVE'])

def main(): 
    preprocessData()
    

# Calling main function
if __name__=="__main__":
     main()

