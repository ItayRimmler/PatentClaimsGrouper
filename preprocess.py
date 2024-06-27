import pandas as pd


def clean(text):
    temp = text['Claim']
    # It seems as if lowcasing the text is enough...
    #import string
    #temp = temp.translate(str.maketrans('', '', string.punctuation))
    temp = temp.lower()
    text['Claim'] = temp
    return text

def split(text):
    temp = text['Claim']
    from nltk.tokenize import word_tokenize
    temp = word_tokenize(temp)
    text['Claim'] = temp
    return text

def stop_words(text):
    from nltk.corpus import stopwords
    if text['Patent'] == '1':
        stop_words = set(stopwords.words('english')).union({'comprising', 'means'})
    elif text['Patent'] == '2':
        stop_words = set(stopwords.words('english')).union({'said'})
    elif text['Patent'] == '3':
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('english'))
    temp = text['Claim']
    temp = [word for word in temp if word not in stop_words]
    text['Claim'] = temp
    return text

def lemmatize(text):
    temp = text['Claim']
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    temp = [lemmatizer.lemmatize(word) for word in temp]
    text['Claim'] = temp
    return text

