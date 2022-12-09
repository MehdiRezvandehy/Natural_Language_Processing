# Required function for running this assignment
# Written by Mehdi Rezvandehy


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import PercentFormatter
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

import nltk
#Tokenization
from nltk.tokenize import word_tokenize
#Stop words
from nltk.corpus import stopwords
stopword_set = set(stopwords.words('english'))
#Stemming 
from nltk.stem import PorterStemmer
#Lemmatization 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##############################################################
def stem(words):
    """Stemming"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def categorize(words):
    tags = nltk.pos_tag(words)
    return [tag for word, tag in tags]

def lemmatize(words, tags):
    """Lemmatization"""
    lemmatizer = WordNetLemmatizer()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    pos = [tag_dict.get(t[0].upper(), wordnet.NOUN) for t in tags]
    return [lemmatizer.lemmatize(w, pos=p) for w, p in zip(words, pos)]

##############################################################

def Conf_Matrix(y_train: [float],y_train_pred:[float], label: [str] ,perfect: int= 0,axt=None,plot: bool =True,
               title: bool =False,t_fontsize: float =8.5,t_y: float=1.2,x_fontsize: float=6.5,
               y_fontsize: float=6.5,trshld: float=0.5) -> [float]:
    
    '''Plot confusion matrix'''
    
    if (y_train_pred.shape[1]==2):
        y_train_pred=[0 if y_train_pred[i][0]>trshld else 1 for i in range(len(y_train_pred))]
    elif (y_train_pred.shape[1]==1):
        y_train_pred=[1 if y_train_pred[i][0]>trshld else 0 for i in range(len(y_train_pred))] 
    else:    
        y_train_pred=[1 if i>trshld else 0 for i in y_train_pred]       
    conf_mx=confusion_matrix(y_train,y_train_pred)
    acr=accuracy_score(y_train,y_train_pred)
    conf_mx =confusion_matrix(y_train,y_train_pred)
    prec=precision_score(y_train,y_train_pred) # == TP/(TP+FP) 
    reca=recall_score(y_train,y_train_pred) # == TP/(TP+FN) ) 
    TN=conf_mx[0][0] ; FP=conf_mx[0][1]
    spec= TN/(TN+FP)        
    if(plot):
        ax1 = axt or plt.axes()
        
        if (perfect==1): y_train_pred=y_train
        
        x=[f'Predicted {label[0]}', f'Predicted {label[1]}']; y=[f'Actual {label[0]}', f'Actual {label[1]}']
        ii=0 
        im =ax1.matshow(conf_mx, cmap='jet', interpolation='nearest') 
        for (i, j), z in np.ndenumerate(conf_mx): 
            if(ii==0): al='TN= '
            if(ii==1): al='FP= '
            if(ii==2): al='FN= '
            if(ii==3): al='TP= '          
            ax1.text(j, i, al+'{:0.0f}'.format(z), color='w', ha='center', va='center', fontweight='bold',fontsize=6.5)
            ii=ii+1
     
        txt='$ Accuracy\,\,\,$=%.2f\n$Sensitivity$=%.2f\n$Precision\,\,\,\,$=%.2f\n$Specificity$=%.2f'
        anchored_text = AnchoredText(txt %(acr,reca,prec,spec), loc=10, borderpad=0)
        ax1.add_artist(anchored_text)    
        
        ax1.set_xticks(np.arange(len(x)))
        ax1.set_xticklabels(x,fontsize=x_fontsize,y=0.97, rotation='horizontal')
        ax1.set_yticks(np.arange(len(y)))
        ax1.set_yticklabels(y,fontsize=y_fontsize,x=0.035, rotation='horizontal') 
        
        cbar =plt.colorbar(im,shrink=0.3,
                           label='Low                              High',orientation='vertical')   
        cbar.set_ticks([])
        plt.title(title,fontsize=t_fontsize,y=t_y)
    return acr, prec, reca, spec 

############################################################

def data_processing(corpus,stopwords=True, Stemming=False,Lemmatization=True):
    
    # Tokenization
    tokens = [word_tokenize(doc.lower()) for doc in corpus]
    
    # Remove punctuation (e.g. ',','.')
    words = [[word for word in tokens_ if word.isalnum()] for tokens_ in tokens] 
    
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos = [categorize(words_) for words_ in words ]
        words  = [set(lemmatize(words_, words_pos_)) for words_, words_pos_ in zip(words,words_pos) ]        
        
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words = [[word for word in tokens_ if word not in stopword_set] for tokens_ in words ]
    
    # Stemming (remove affixes)
    if Stemming:
        words_doc = [stem(tokens_) for tokens_ in words ]

    # Prepare corpose CountVectorizer
    words=[" ".join(i) for i in words]
    
    return words

############################################################

def CountVectorizer_train(train, test=False, counts=True, Tfidf=False): 
    
    words_tarin=data_processing(train)
    
    # if TF-idf is applied
    if Tfidf:
        vect = TfidfVectorizer(min_df=1)
        sparse_matrix = vect.fit_transform(words_tarin)
        doc_term_matrix_train = sparse_matrix.todense()
    else:
        vect = CountVectorizer()
        sparse_matrix = vect.fit_transform(words_tarin)
        doc_term_matrix_train = sparse_matrix.todense()        
    #     
    if (counts ):
        df_counts = pd.DataFrame(doc_term_matrix_train, 
                          columns=vect.get_feature_names())
        return doc_term_matrix_train, df_counts, vect
    else:
        return doc_term_matrix_train, vect

############################################################

def CountVectorizer_test(test, model, counts=True, Tfidf=False): 
    
    words_test=data_processing(test)
    sparse_matrix = model.transform(words_test)
    doc_term_matrix_test = sparse_matrix.todense()  
 
    #     
    if (counts ):
        df_counts = pd.DataFrame(doc_term_matrix_test, 
                          columns=vect.get_feature_names())
        return doc_term_matrix_test, df_counts
    else:
        return doc_term_matrix_test

############################################################

def AUC(prediction: [float],y_train: [float], n_algorithm: int
       ,label:[str],title: str='Receiver Operating Characteristic (ROC)'
       ,linewidth=2) -> None:
    
    '''Plot Receiver Operating Characteristic (ROC) for predictors'''
    
    color=['b','r','g','y','m','c']
    for i in range(n_algorithm):
        fpr, tpr, thresold = roc_curve(y_train, prediction[i][:,1])
        roc_auc = auc(fpr, tpr)
        if (i==0):
            tmp_linewidth=4
            cm='k--'
        else:
            tmp_linewidth=linewidth
            cm= f'{color[i]}-'
            
        plt.plot(fpr, tpr,cm, linewidth=tmp_linewidth,
                 label=label[i]+' (AUC =' + r"$\bf{" + str(np.round(roc_auc,3)) + "}$"+')')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate (1-Specificity) FP/(FP+TN)',fontsize=12)
    plt.ylabel('True Positive Rate (Sensistivity) TP/(TP+FN)',fontsize=12)
    plt.title(title,fontsize=15)
    plt.grid(linewidth='0.25')
    plt.legend(loc="lower right",fontsize=11)
    plt.show()         

