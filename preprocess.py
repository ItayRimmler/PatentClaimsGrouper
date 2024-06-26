def clean(text):
    text = text[1]
    import string
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def split(text):
    text = text[1]
    from nltk.tokenize import word_tokenize
    text = word_tokenize(text)
    return text

def stop_words(text):
    from nltk.corpus import stopwords
    if text[0] == '1':
        stop_words = set(stopwords.words('english'))
    elif text[0] == '2':
        stop_words = set(stopwords.words('english'))
    elif text[0] == '3':
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = set(stopwords.words('english'))
    text = text[1]
    text = [word for word in text if word not in stop_words]
    return text

def lemmatize(text):
    text = text[1]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return text

