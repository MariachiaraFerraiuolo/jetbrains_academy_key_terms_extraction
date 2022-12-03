import string
import pandas as pd
import nltk
import numpy as np
import sklearn
from lxml import etree
from collections import Counter
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

el_stopwords = list(stopwords.words('english')) + ['ha', 'wa', 'u', 'a', "le"]
punctuation = string.punctuation
list_of_headlines= []
list_of_news = []
news_lowered = []
list_of_nouns = []
string_list= []
vectorizer = TfidfVectorizer()
xml_path = "news.xml"
root = etree.parse(xml_path).getroot()

for i in range(len(root[0])):
    list_of_headlines.append(root[0][i][0].text)
for k in range(len(root[0])):
    list_of_news.append(root[0][k][1].text)


def process_text(text):
    lower_text = text.lower()
    tokens = nltk.tokenize.word_tokenize(lower_text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens_without_sw = [word for word in lemmatized_tokens if not word in el_stopwords and not word in punctuation]
    pos_speech = [nltk.pos_tag([word]) for word in tokens_without_sw]
    temp_list = [word[0][0] for word in pos_speech if word[0][1] == 'NN']
    return temp_list

for news in list_of_news:
    news_lowered.append(process_text(news))

for i in range(0, len(news_lowered)):
    new_string = ', '.join(news_lowered[i])
    string_list.append(new_string)

tfidf_matrix = vectorizer.fit_transform(string_list)
terms = vectorizer.get_feature_names_out()

for i in range(len(list_of_headlines)):
    list_of_words = []
    print(list_of_headlines[i]+':')
    df = pd.DataFrame(tfidf_matrix[i].toarray())
    df2 = df.transpose().sort_values(by=0, ascending=False).reset_index()
    for i in range(0, len(df2)):
        k = terms[df2['index'][i]]
        list_of_words.append(k)
    df2['words'] = list_of_words
    df2 = df2.sort_values(by=[0,'words'], ascending=[False, False])
    string_to_print = ''
    for n in range(0, 5):
       part = df2.iloc[n]['words']
       string_to_print += part + ' '
    print(string_to_print + '\n')



