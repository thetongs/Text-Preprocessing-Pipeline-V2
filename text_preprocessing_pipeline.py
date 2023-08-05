import string
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemer = PorterStemmer()
lemma = WordNetLemmatizer()

from emo_unicode import EMOTICONS
from chat_words import SHORT_FORMS
from unicode_emo import UNICODE_EMO

from autocorrect import Speller
speller = Speller()

import contractions
import wordninja


# text preprocessing pipeline
def text_preprocessing(
    data,
    lower_case = True,
    expand_form = True,
    separate_words = True,
    punc = True,
    stopwards = False,
    stem = False,
    lemmat = True,
    urls = True,
    htmltags = True,
    remove_num_spaces = True,
    spell_chec = True,
    rm_emoj = False,
    rm_emticon = False,
    cn_emticon = True,
    cn_emoj = True,
    chat_con = True,
):
    # change to lowercase
    if(lower_case):
        data = data.lower()

    # remove html tags
    if(htmltags):
        html_pattern = re.compile("<.*?>")
        data = html_pattern.sub(r'', data)

    # remove urls
    if(urls):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        data = url_pattern.sub(r'', data)

    # remove emojis
    if(rm_emoj):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        data = emoji_pattern.sub(r'', data)

    # remove emoticons
    if(rm_emticon):
        emoticon_pattern = re.compile(u'(' + u'|'.join(k for k in EMOTICONS) + u')')
        data = emoticon_pattern.sub(r'', data)

    # convert emojis to words
    if(cn_emoj):
        for emot in UNICODE_EMO:
            data = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()), data)

    # convert emoticons to words
    if(cn_emticon):
        for emot in EMOTICONS:
            data = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",", "").split()), data)

    # contraction to expanded form
    if(expand_form):
        data = contractions.fix(data)

    # chat words conversion
    if(chat_con):
        text = " "
        for w in data.split():
            if w in SHORT_FORMS.keys():
                text = text + " " + SHORT_FORMS[w]
            else:
                text = text + " " + w
        data = text

    # spelling checker
    if(spell_chec):
        data = speller(data)

    # seperate combined words
    if(separate_words):
      data = " ".join(wordninja.split(data))

    # remove stopwords
    if(stopwards):
        data = ' '.join([word for word in word_tokenize(data) if not word in list(stopwords.words('english'))])

    # stemming
    if(stem):
        data = ' '.join([stemer.stem(word) for word in word_tokenize(data)])

    # lemmatization
    if(lemmat):
        data = ' '.join([lemma.lemmatize(word) for word in word_tokenize(data)])

    # remove punctuations
    if(punc):
        data = data.translate(str.maketrans('', '', string.punctuation))

    # remove numbers and extra spaces
    if(remove_num_spaces):
        pattern = r"\d+"
        data = re.sub(pattern, "", str(data))
        data.replace('\s+', ' ')

    return data

data = """Cake is a form of sweet food made from flour, sugar, and other ingredients, that is usually baked.
In their oldest forms, cakes were modifications of bread, but caakes now cover a wide range of preparations
that can be simple or elaborate, and that share features with other desserts such as pastries, meringues, custards,
and pies BRB :-) 🔥. thisisatestsentence"""

data_ = text_preprocessing(data)
data_

# dataset['preprocessed_data'] = dataset['text'].apply(func = lambda x:text_preprocessing(x))
# dataset.head()

