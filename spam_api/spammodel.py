import pandas as pd
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# use your file path of csv file
df = pd.read_csv('C:/Users/sko98/Downloads/archive (3)/spam_ham_dataset.csv')  



df['text'] = df['text'].str.replace('\n','')
df['text'] = df['text'].str.replace('\r','')
df['text'] = df['text'].str.replace('{', '')
df['text'] = df['text'].str.replace('}', '')
df['text'] = df['text'].str.replace('Subject:', '')
df['text'] = df['text'].str.replace(':','')
df['text'] = df['text'].str.replace(';','')
df['text'] = df['text'].str.replace('.','')
df['text'] = df['text'].str.replace(r'\s{2,}', '')


# remove the  punctuation

import string

for c in string.punctuation:
    df['text'] = df['text'].str.replace(c,'')



# remove the stopword


nltk.download('stopwords')
remove_stopwords = lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words('english'))])
df['text'] = df['text'].apply(remove_stopwords)





# Create a TfidfVectorizer object

vectorizer = TfidfVectorizer()



X = vectorizer.fit_transform(df['text'])

y = df['label_num']
model = LogisticRegression()

model.fit(X,y)


import re

def clean_text(text):
    
    # List of replacements (old string, new string, and optional regex flag)
    replacements = [
        ('\n', ''), 
        ('\r', ''), 
        ('{', ''), 
        ('}', ''), 
        ('Subject:', ''), 
        (':', ''), 
        (';', ''), 
        ('.', ''), 
        (r'\s{2,}', ' ', True)  # Regex to replace multiple spaces with a single space
    ]
    
    # Apply replacements
    for old, new, *is_regex in replacements:
        regex = is_regex[0] if is_regex else False
        if regex:
            text = re.sub(old, new, text)
        else:
            text = text.replace(old, new)
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = remove_stopwords(text) 
    
    return text
    


def predictmodel(text):
    clean_data = clean_text(text)
    X_text = vectorizer.transform([clean_data])
    
    result = model.predict(X_text)
    
    return result[0]
    
    
