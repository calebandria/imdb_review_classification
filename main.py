from text_classification_tools import BagOfWords, TfIdf, Ngram
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec


def utilities_data(dataPath, labelCol="sentiment"):
    dataset = pd.read_csv(dataPath)
    dataset.head()
    dataset.shape
    # trying to look at the values of the sentiment
    print("Category of sentiment in the dataset: ", dataset[labelCol].unique())
    dataset.describe()
    #sentiment count
    dataset[labelCol].value_counts()

    train_data, test_data= train_test_split(dataset, test_size=0.2, random_state=42)
    print("The size ot the train dataset is: ", train_data.shape)
    print("The size ot the test dataset is: ", test_data.shape)

    stopword_list=nltk.corpus.stopwords.words('english')
    tokenizer = ToktokTokenizer()

    return train_data, test_data, tokenizer, stopword_list

def main():
    train_data, test_data, tokenizer, stopword_list = utilities_data('./data/IMDB Dataset.csv')

    #for bag of words and then logistic regression

    """ bow = BagOfWords(tokenizer,stopword_list)
    vectorizer, classifier = bow.modelLogisticRegressionTraining(train_data)
    bow.modelLogisticRegressionTesting(test_data, vectorizer, classifier) """

    #for tfidf and then logistic regression

    """ tfidf = TfIdf(tokenizer, stopword_list)
    vectorizer, classifier = tfidf.modelLogisticRegressionTraining(train_data)
    tfidf.modelLogisticRegressionTesting(test_data, vectorizer, classifier)
 """

    #for ngram=2 
    ngram_2 = Ngram(tokenizer, stopword_list, 2)
    vectorizer, classifier = ngram_2.modelLogisticRegressionTraining(train_data)
    ngram_2.modelLogisticRegressionTesting(test_data, vectorizer, classifier)

    #for ngram=3
    ngram_3 = Ngram(tokenizer, stopword_list, 3)
    vectorizer, classifier = ngram_3.modelLogisticRegressionTraining(train_data)
    ngram_3.modelLogisticRegressionTesting(test_data, vectorizer, classifier)

    #for ngram = 5
    ngram_5 = Ngram(tokenizer, stopword_list, 5)
    vectorizer, classifier = ngram_5.modelLogisticRegressionTraining(train_data)
    ngram_5.modelLogisticRegressionTesting(test_data, vectorizer, classifier)   
    

if __name__ == "__main__":
    main()