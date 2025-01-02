from sklearn.feature_extraction.text import TfidfVectorizer

class TfIdf(BagOfWords):
  def __init__(self,tokenizer,stopword_list,reviewCol="review",sentimentCol="sentiment"):
    super().__init__(tokenizer,stopword_list)

  def feature_extraction(self,data):
    cv = TfidfVectorizer()
    data_X, data_y = self.final_version_data(data)
    cv_train_data = cv.fit_transform(data_X)
    #cv_test_data = cv.transform(test_review)

    print("Train shape: ", cv_train_data.shape)
    #print("Test shape: ", cv_test_data.shape)
    
    return cv, cv_train_data, data_y

