import warnings
warnings.filterwarnings("ignore")
import pickle
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

#doc_new = ['obama is running for president in 2016']

var = input("Please enter the news text you want to verify: ")

stop = stopwords.words('english')
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    words = [lemmatizer.lemmatize(w) for w in word_tokenize(text)]
    return " ".join(words)

def remove_stop(text):
    words = [word for word in word_tokenize(text) if not word in stopwords.words()]
    return " ".join(words)

def wordopt(text):
    text = text.lower()
    text = re.sub('[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n\n', '', text)
    return text

#function to run for prediction
def detecting_fake_news(var):
#retrieving the best model for prediction call
    load_model = pickle.load(open('ensemble_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("The given statement is ",prediction[0]),
        print("The truth probability is ",prob[0][1]))


if __name__ == '__main__':


    var = remove_stop(var)
    var = lemmatize_text(var)
    var = wordopt(var)

    print("You entered: " + str(var))

    detecting_fake_news(var)




