import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from numpy.linalg import norm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

def getReview(review):
    review_result = "none"
    review = review.lower()
    review = re.sub('[^A-Za-z]+', ' ', review)
    sentiment_dict = sid.polarity_scores(review.strip())
    compound = sentiment_dict['compound']
    if compound >= 0.05 : 
        review_result = 'Positive'
    return review_result    
    
    
dataset = pd.read_csv("Dataset/amazon_reviews.csv")
dataset = dataset.values
text = dataset[:,0]
label = dataset[:,1]

tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=1000,lowercase=True)
tfidf = tfidf_vectorizer.fit_transform(text).toarray()        
df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
print(df.shape)
df = df.values
X = df[:, 0:1000]

testData = ['DVD','CD','Phone']
for i in range(len(testData)):
    test = testData[i].lower()
    test = tfidf_vectorizer.transform([test]).toarray()
    test = test[0]
    similarity = 0
    review = 'none'
    rating = 0
    for j in range(len(X)):
        review_score = dot(X[j], test)/(norm(X[j])*norm(test))
        if review_score > similarity:
            similarity = review_score
            review_type = getReview(text[j])
            if review_type == 'Positive':
                review = text[j]
                rating = label[j]
    print(str(review)+"==="+str(rating))            
        
        


