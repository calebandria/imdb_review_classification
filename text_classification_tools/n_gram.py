from .bag_of_word import BagOfWords
from sklearn.feature_extraction.text import CountVectorizer

class Ngram(BagOfWords):
    def __init__(self,tokenizer,stopword_list,ngram, reviewCol="review",sentimentCol="sentiment"):
    super().__init__(tokenizer,stopword_list)
    self ngram = ngram

    def feature_extraction(self,data):
        cv = CountVectorizer(ngram_range=(self.ngram, self.ngram), binary=True)
        data_X, data_y = self.final_version_data(data)
        cv_train_data = cv.fit_transform(data_X)
        #cv_test_data = cv.transform(test_review)

        print("Train shape: ", cv_train_data.shape)
        #print("Test shape: ", cv_test_data.shape)
        
        return cv, cv_train_data, data_y