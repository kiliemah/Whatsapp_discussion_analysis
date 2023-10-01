import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import string
import pickle


from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()

from keras.models import load_model

nltk.download('punkt')




# Defining preprocessing functions

def lower_function(data):
    """
    Take the training data frame and lower all messages 
    """
    result = data.str.lower()
    return result


def deEmojify(data):
    """
    Remove emoji from messages 
    """
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags = re.UNICODE)
    
    def edit(raw):
        return regrex_pattern.sub(r'',raw)
    
    result = data.apply(edit)
    return result
  
        
    
def strip_links(data):
    """
    Remove links in comments
    """
    def edit(raw):
        
        link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
        links         = re.findall(link_regex, raw)
        for link in links:
            raw = raw.replace(link[0], ', ')

        raw = raw.replace("…",' ')
        raw = raw.replace("’"," " )
        raw = raw.replace("'"," ")
        return raw
    
    result = data.apply(edit)
    return result


   
def special_caractere_delete(data):
    """
    Special strings deletion 
    """
    def strip_all_entities(raw):
        #text=re.sub('[^A-Za-z0-9]+', ' ', text)
        entity_prefixes = ['@','#',"_","'","’"]
        for separator in string.punctuation:
            if separator not in entity_prefixes :
                raw = raw.replace(separator,' ')
        words = []
        for word in raw.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        r = ' '.join(words)
        return r
    
    result = data.apply(strip_all_entities)
    return result


def data_preprocessing(data):
    result = lower_function(data)
    print("--- lower : OK")
    result = deEmojify(result)
    print("--- Removing Emoji : OK")
    result = strip_links(result)
    print("--- Removing links : OK")
    result = special_caractere_delete(result)
    print("--- Removing special characters : OK")
    return result

# tokenizing dataset
def Tokenization(data,num_word):
    """
    Building a set of words based on the dataset. The size is specified by num_word
    """
    tokenizer=Tokenizer(num_word,lower=True)
    tokenizer.fit_on_texts(data)
    return tokenizer


def empty_word_deletes(data):
        
    result = data.apply(word_tokenize)
    return result
        




def load_train_data(file):
    data=pd.read_csv(file)
    data["comment_text"]=data["comment_text"].astype("str")
    return(data)
    
    
def load_whatsapp_conversation(file):
    with open(file, encoding='utf8') as f:
        lines = f.readlines()
        #Supression du caractère spécial (‎)
        i=0
        for line in lines:
            if "‎" in line:
                lines[i] = lines[i].replace("‎","")
            i=i+1
        # Création d'un nouveau fichier temp, sans le caractère spécial
        with open('temp.txt', 'w',encoding='utf8') as f_upgrade:
            for item in lines:
                f_upgrade.write("%s" % item)
            
    date=[]
    interlocuteur=[]
    text=[]
    
    # Extraction des données
    with open('temp.txt','r',encoding='utf8') as tf:
        lines = tf.read().split('\n[')

    i=0
    for line in lines:

        split=re.split('] ', line.strip())
        date.append(split[0])
        final_split=re.split(': ', split[1])
        interlocuteur.append(final_split[0])
        text.append(final_split[1])

        i=i+1
    # Treatment of date column
    for val in date: 
        if val[0]=='[':
            val = val.replace(val[0],"")
        
    
    # Bulding resulting dataFrame
    dict={'date':date,'person':interlocuteur,'Message':text}
    data=pd.DataFrame(dict)
    return (data)

def main(whatsapp_file):
    
    Discussion = load_whatsapp_conversation("Discussion_file/_chat.txt")
    original_message = Discussion['Message']
    Discussion['Message'] = data_preprocessing(Discussion['Message'])
    # Load data from the pickle file
    with open('../Tokenizer/tokenizer.pickle', 'rb') as file:
        tokenizer = pickle.load(file)
        
    with open('../Models/best_threshold.pkl','rb') as file:
        thr = pickle.load(file)
    
    
    sequences=tokenizer.texts_to_sequences(Discussion["Message"])
    sequences_pad=pad_sequences(sequences,maxlen=200,padding='post')
    model = load_model("../Models/model.h5")

    
    result = model.predict(sequences_pad)
    prediction=[]
    proba=[]
    for pred in result:
        for y in pred:
            if y>0.97:
                prediction.append("positif")
                proba.append(y)
            elif y<0.97:
                prediction.append("negatif")
                proba.append(y)
            else:
                prediction.append("neutre")
                proba.append(y)

    prediction_dict={'prediction':prediction,'proba':proba}
    prediction_df=pd.DataFrame(prediction_dict)
    
    final_df = pd.concat([Discussion,prediction_df],axis=1)
    final_df['Message'] = original_message
    return final_df[final_df["prediction"]=="positif"]
    
    