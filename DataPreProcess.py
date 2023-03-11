import pandas as pd
import numpy as np
import string as st
import re
import nltk
from matplotlib import pyplot as plt
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# test_filename = 'test.csv'
# train_filename = 'train.csv'
# # valid_filename = 'valid.csv'
#
# train_data_short = pd.read_csv(train_filename)
# test_data_short = pd.read_csv(test_filename)
# # valid_data = pd.read_csv(valid_filename)
#
# train_data_short = pd.concat([pd.read_csv(train_filename), pd.read_csv(test_filename)], ignore_index=True)


# false_dataframe = pd.read_csv('Fake.csv')
# false_dataframe['label'] = 0
# true_dataframe = pd.read_csv('True.csv')
# true_dataframe['label'] = 1
# Contact false & true dataframe
# train_df = pd.concat([false_dataframe, true_dataframe], ignore_index=True)
# Drop not required columns
# train_df = train_df.drop(["title", "subject", "date"], axis=1)

train_df = pd.read_csv('WELFake_Dataset.csv')
train_df = train_df.drop(["title"], axis=1)

# train_df = train_data_short
train_df = train_df.sample(frac=1, random_state=1).reset_index(drop=True)

# remove row with null text and duplicate row
train_df.dropna(axis=0, how='any', inplace=True)
train_df.drop_duplicates(keep='first', inplace=True)

def content_length(cell):
    return len(str(cell))

rows_to_drop = train_df.index[train_df['text'].apply(content_length) < 50].tolist()
train_df.drop(index=rows_to_drop, inplace=True)

def drop_prefix(text, prefix, n=5):
    ts = str.split(text, ' ')
    if prefix in ts[:n]:
        return str.split(text, prefix)[-1]
    else:
        return text

# remove prefix include Reuters && delete punctuation and number
train_df["text"] = train_df["text"].apply(lambda x: drop_prefix(x, '(Reuters)'))
train_df["text"] = train_df["text"].str.replace('[^\w\s]', '')
train_df["text"] = train_df["text"].str.replace('\d+', '')

# remove url in text column
train_df['text'] = train_df['text'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
train_df['text'].replace('', np.nan, inplace=True)

# convert text content to lower case
train_df['text'] = train_df['text'].apply(lambda x: x.lower())

# remove stop words
stop = stopwords.words('english')
train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    words = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)]
    return " ".join(words)

train_df['text'] = train_df['text'].apply(lemmatize_text)

# text = train_df['text'].values
# from wordcloud import WordCloud
# wc = WordCloud(width = 1000 , height = 500).generate(str(text))
# wc.to_file(r"C:\Users\OY1\PycharmProjects\wordcloud.png")

# def word_stemming(text):
#     word_stemmer = PorterStemmer()
#     return [word_stemmer.stem(word) for word in text if word not in stopwords.words('english')]
#
# min_df['text'] = min_df['text'].apply(lambda x: word_stemming(x))

print(train_df['text'])
print(train_df.head(100))
print(train_df.info())
