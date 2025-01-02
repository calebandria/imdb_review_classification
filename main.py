from text_classification_tools import BagOfWords, TfIdf
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec


def main():
    dataset = pd.read_csv('./data/IMDB Dataset.csv')
    dataset.head()
    dataset.shape
    # trying to look at the values of the sentiment
    print("Category of sentiment in the dataset: ", dataset["sentiment"].unique())
    dataset.describe()
    #sentiment count
    dataset['sentiment'].value_counts()

    train_data, test_data= train_test_split(dataset, test_size=0.2, random_state=42)
    print("The size ot the train dataset is: ", train_data.shape)
    print("The size ot the test dataset is: ", test_data.shape)

    stopword_list=nltk.corpus.stopwords.words('english')
    tokenizer = ToktokTokenizer()

    #for bag of words and then logistic regression
    #bow = BagOfWords(tokenizer,stopword_list)
    #vectorizer, classifier = bow.modelLogisticRegressionTraining(train_data)
    #bow.modelLogisticRegressionTesting(test_data, vectorizer, classifier)

    #for tfidf and then logistic regression
    tfidf = TfIdf(tokenizer, stopword_list)
    vectorizer, classifier = tfidf.modelLogisticRegressionTraining(train_data)
    tfidf.modelLogisticRegressionTesting(test_data, vectorizer, classifier)

    

if __name__ == "__main__":
    main()