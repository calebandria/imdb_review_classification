import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Download required nltk resource
nltk.download('stopwords')
#nltk.download('punkt')
nltk.download('wordnet')

class BagOfWords:
  def __init__(self,tokenizer,stopword_list,reviewCol="review",sentimentCol="sentiment"):
    #self.data = data
    #self.features = features
    self.stopword_list = stopword_list
    self.tokenizer = tokenizer
    self.reviewCol = reviewCol
    self.sentimentCol = sentimentCol

  #removing the html strips
  def strip_html(self,text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

  #removing all the special characters
  def remove_special_characters(self,text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

  def remove_between_square_brackets(self,text):
    return re.sub('\[[^]]*\]', '', text)

  #Removing the noisy text
  def denoise_text(self,text):
      text = self.strip_html(text)
      text = self.remove_between_square_brackets(text)
      text = self.remove_special_characters(text)
      return text

  #Stemming the text
  def simple_stemmer(self,text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

  def lemmatizer(self,text):
    ps = WordNetLemmatizer()
    text = ' '.join([ps.lemmatize(word) for word in text.split()])
    return text


  #removing the stopwords
  def remove_stopwords(self, text,is_lower_case=False):
      tokens = self.tokenizer.tokenize(text)
      tokens = [token.strip() for token in tokens]
      if is_lower_case:
          filtered_tokens = [token for token in tokens if token not in self.stopword_list]
      else:
          filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list]
      filtered_text = ' '.join(filtered_tokens)
      return filtered_text

  def final_version_data(self, data):
    data[self.reviewCol] = data[self.reviewCol].apply(self.denoise_text)
    data[self.reviewCol] = data[self.reviewCol].apply(self.lemmatizer)
    data[self.reviewCol] = data[self.reviewCol].apply(self.simple_stemmer)
    data[self.reviewCol] = data[self.reviewCol].apply(self.remove_stopwords)

    data[self.sentimentCol] = data[self.sentimentCol].map({'positive': 1, 'negative': 0})

    print("Number of NaN values in y: ", {np.isnan(data[self.sentimentCol]).sum()})
    return data[self.reviewCol], data[self.sentimentCol]


  def feature_extraction(self,data):
    cv = CountVectorizer(binary=True)
    # train test split
    data_X, data_y = self.final_version_data(data)
    #train_review, test_review, train_label, test_label = train_test_split(data_X, data_y, test_size=0.2, random_state=42)
    cv_train_data = cv.fit_transform(data_X)
    data_y = data_y.astype('int')

    print("Number of NaN values in y after type conversion: ", {np.isnan(data_y).sum()})
    #cv_test_data = cv.transform(test_review)

    print("Train shape: ", cv_train_data.shape)
    
    #print("Test shape: ", cv_test_data.shape)


    return cv, cv_train_data, data_y

  def modelLogisticRegressionTraining(self,training_data):
    clf = LogisticRegression()

    #training
    vectorizer, X_train, y_train = self.feature_extraction(training_data)
    clf.fit(X_train,y_train)

    y_train_pred = clf.predict(X_train)

    training_accuracy = clf.score(X_train, y_train)
    cm_matrix_train = confusion_matrix(y_train, y_train_pred )
    class_report_train = classification_report(y_train, y_train_pred)


    print(f"Training: \nAccuracy: {training_accuracy} \nConfusion matrix: \n{cm_matrix_train} \n All other classification metrics: \n{class_report_train}")

    return vectorizer, clf

  def modelLogisticRegressionTesting(self, test_data, vectorizer, classifier):

    data_test_X, y_test = self.final_version_data(test_data)

    #vectorizing test data
    #vectorizer, classifier = self.modelLogisticRegressionTraining()
    X_test = vectorizer.transform(data_test_X)

    print("Test shape: ", X_test.shape)

    #testing
    y_pred = classifier.predict(X_test)

    print("where is the wrong thing")
    #score evaluation
    test_accuracy = accuracy_score(y_test, y_pred)
    cm_matrix_test = confusion_matrix(y_test, y_pred)
    class_report_test = classification_report(y_test, y_pred)

    print(f"Test: \nAccuracy: {test_accuracy} \nConfusion matrix: \n{cm_matrix_test} \n All other classification metrics: \n{class_report_test}")

