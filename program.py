import pandas as pd
import numpy as np
import nltk

from nltk.tokenize import word_tokenize
import string

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

from nltk.corpus import stopwords

import matplotlib.pyplot as plt
plt.style.use('ggplot')

my_punc = ['!','$','%','?']

string.punctuation = string.punctuation + '--'

def prepare_text(text):         # Text cleaning and Preprocessing
    text.replace("\'","")
    text = text.lower()
    
    tok_words = word_tokenize(text)
    
    non_punc = []
    for i in tok_words:
        if i in string.punctuation and i not in my_punc:
            pass
        else:
            non_punc.append(i)
    
    non_stop = []
    stop_words = stopwords.words("english")
    for i in non_punc:
       if i not in stop_words:
           non_stop.append(i)
    
    return ' '.join(non_stop)

# Reading data into a Pandas DataFrame

data = pd.read_csv('finanical_news_dataset.csv',encoding='latin-1')
data = data.rename(columns = {'Label':'Y','CN':'text'})
data.text.str.encode('utf-8')

da = data

for i in da.index:
    da.loc[i,'text'] = prepare_text(da.loc[i,'text'])

# Furthur Preprocessing of text  -  rare words removed, words lemmatized

freq_rare = pd.Series(' '.join(da.text).split()).value_counts()[-20:]
freq_rare = list(freq_rare.index)
copy = da.text
copy = copy.apply(lambda x : " ".join(x for x in x.split() if x not in freq_rare))

from textblob import Word
copy = copy.apply(lambda x : " ".join([Word(w).lemmatize() for w in x.split()]))

da['text'] = copy

def print_results(name,y_true,y_pred1,y_pred2):
    print("\n",name," ROC_AUC : ",roc_auc_score(y_true, y_pred2))
    print("\n",name," Model Accuracy : ",accuracy_score(y_true, y_pred1))
    print("\n",name,"Confusion Matrix : \n")
    print(confusion_matrix(y_true, y_pred1))
    print("\nMetrics : \n",precision_recall_fscore_support(y_true, y_pred1))
    print("\nClassification Report : \n",classification_report(y_true, y_pred1))

def models(x_train,x_test,y_train,y_test,algo):         # Testing with ML Algorithms

    if algo == 1:
        from sklearn import naive_bayes
        clf = naive_bayes.MultinomialNB()
        clf.fit(x_train,y_train)            
        print_results('Naive Bayes',y_test,clf.predict(x_test),clf.predict_proba(x_test)[:,1])
        
    elif algo == 2:
        from sklearn import svm
        clf = svm.SVC(kernel = "linear",probability = True)
        clf = clf.fit(x_train,y_train)        
        print_results('SVM',y_test,clf.predict(x_test),clf.predict_proba(x_test)[:,1])

    elif algo == 3:       
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        clf.fit(x_train,y_train)
        print_results('Log Reg',y_test,clf.predict(x_test),clf.predict_proba(x_test)[:,1])

    elif algo == 4:
        from sklearn import tree
        clf = tree.DecisionTreeClassifier(criterion = "entropy")
        clf.fit(x_train, y_train)
        print(clf.feature_importances_)
        print_results('DTree',y_test,clf.predict(x_test),clf.predict_proba(x_test)[:,1])

    elif algo == 5:
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier()
        clf = clf.fit(x_train,y_train)
        print_results('KNN',y_test,clf.predict(x_test),clf.predict_proba(x_test)[:,1])
    
    elif algo == 6:        
        from sklearn.cluster import KMeans         #Look at the Example... This is wrong
        clf = KMeans(n_clusters=2).fit(x_train)
        print(clf.predict(x_test))
        print_results('K Means',y_test,clf.predict(x_test),clf.predict(x_test))
    
    else:
        print("Invalid Option \n")
        
##################          

sx_train, sx_test, sy_train, sy_test = train_test_split(da['text'],da['Y'],train_size = 0.7)

bow = CountVectorizer()     # Bag Of Words model
bow.fit(sx_train)

bow_X_train = bow.transform(sx_train)
bow_X_test = bow.transform(sx_test)
bow_Y_train = sy_train
bow_Y_test = sy_test

models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,1)
models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,2)
models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,3)
models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,4)
models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,5)
models(bow_X_train, bow_X_test, bow_Y_train, bow_Y_test,6)

tfidf = TfidfVectorizer()       # Tf Idf Model
tfidf.fit(sx_train)

tfidf_X_train = tfidf.transform(sx_train)
tfidf_X_test = tfidf.transform(sx_test)
tfidf_Y_train = sy_train
tfidf_Y_test = sy_test

models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,1)
models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,2)
models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,3)
models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,4)
models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,5)
models(tfidf_X_train, tfidf_X_test, tfidf_Y_train, tfidf_Y_test,6)

##################

## Bigram Model 

tfidf_ngram = TfidfVectorizer(analyzer = 'word', ngram_range = (2,2))
tfidf_ngram = tfidf_ngram.fit(sx_train)
ngram_X_train = tfidf_ngram.transform(sx_train)
ngram_X_test = tfidf_ngram.transform(sx_test)
ngram_Y = sy_train

models(ngram_X_train, ngram_X_test, tfidf_Y_train, tfidf_Y_test,1)

##xx Bigram Model

                  ######## Word2Vec

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
from gensim import models

wv_x = da['text'].values.tolist()

wv_corp = wv_x 
wv_corp_tok = [nltk.word_tokenize(str(sent)) for sent in wv_corp]

wv_mod = gensim.models.Word2Vec(wv_corp_tok, min_count = 2, size = 70, workers = 10)      # min_count --> if prerent atlleast once we include it
wv_mod.train(wv_corp_tok,total_examples = len(wv_corp_tok),epochs = 10)

                #########  Keras

from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Activation, Conv2D

vc = len(set(nltk.word_tokenize(str(list(da.text)))))
    
## Vectorising by 1-hot encoding

oh_X_train = [one_hot(sx_train[i], vc) for i in sx_train.index]
oh_X_test = [one_hot(sx_test[i], vc) for i in sx_test.index]

oh_X_len = max(max([len(i) for i in oh_X_train]) , max([len(i) for i in oh_X_test]))      # length of maximum list in list of lists (for pad length)

oh_X_train_pad = pad_sequences(oh_X_train, oh_X_len, padding = 'post')
oh_X_test_pad = pad_sequences(oh_X_test, oh_X_len, padding = 'post')

##xx Vectorising by 1-hot encoding

## Vectorising by Tokenizer()

tok = Tokenizer(num_words = vc)
tok.fit_on_texts(sx_train)

tok_X_train = tok.texts_to_sequences(sx_train)
tok_X_test = tok.texts_to_sequences(sx_test)

vocab_size = len(tok.word_index) + 1

tok_X_len = max(max([len(i) for i in tok_X_train]) , max([len(i) for i in tok_X_test]))

tok_X_train_pad = pad_sequences(tok_X_train, tok_X_len, padding = 'post')
tok_X_test_pad = pad_sequences(tok_X_test, tok_X_len, padding = 'post')

##xx Vectorising by Tokenizer()

def deep1(x_train, x_test, y_train, y_test, vc, mlen):
    model = Sequential()
    model.add(Embedding(round(vc*1.5), 70, input_length = mlen))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    
    model.fit(x = x_train, y = np.array(y_train), epochs = 10, verbose = 2)
    loss, accuracy = model.evaluate(x = x_test, y = np.array(y_test), verbose=2)
    print('Accuracy: %f' % (accuracy*100))
    
    y_pred = model.predict_classes(x_test)
    plot_roc_curve(sy_test, y_pred,'')

deep1(oh_X_train_pad, oh_X_test_pad, sy_train, sy_test, vc, oh_X_len)   
deep1(tok_X_train_pad, tok_X_test_pad, sy_train, sy_test, vocab_size, tok_X_len)


                ############# GloVE pretrained
                
t = Tokenizer()
t.fit_on_texts(sx_train)
vocab_size = len(t.word_index) + 1
tok_X_train = t.texts_to_sequences(sx_train)
tok_X_test = t.texts_to_sequences(sx_test)
tok_X_len = max(max([len(i) for i in tok_X_train]) , max([len(i) for i in tok_X_test]))

tok_X_train_pad = pad_sequences(tok_X_train, maxlen = tok_X_len, padding = 'post')
tok_X_test_pad = pad_sequences(tok_X_test, maxlen = tok_X_len, padding = 'post')

embedding_index = dict()
f = open('glove.6B.100d.txt',encoding = "utf8")
for line in f:
    value = line.split()
    word = value[0]
    weights = np.asarray(value[1:],dtype='float32')            
    embedding_index[word] = weights
f.close()

print('Loaded %s word vectors.' % len(embedding_index))

embedding_matrix = np.zeros((vocab_size,100))
for word, i in t.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not  None:
        embedding_matrix[i] = embedding_vector

# Check how many elements are non-zero
#nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
#print(nonzero_elements / vocab_size)

print("\nUsing GloVe Pretrained Weights\n\n")

model = Sequential()
model.add(Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = tok_X_len, trainable = True))
model.add(GlobalMaxPooling1D())
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(tok_X_train_pad, np.array(sy_train), epochs = 8 )

loss, accuracy = model.evaluate(tok_X_test_pad, np.array(sy_test), verbose = 2)
print('Accuracy: %f' % (accuracy*100))

y_pred = model.predict_classes(tok_X_test_pad)
plot_roc_curve(sy_test, y_pred,'GloVe Pretrained')

print("\nCNN Model Glove and Own\n\n")

model = Sequential()
            #   glove inputs
#model.add(Embedding(vocab_size, 100, weights = [embedding_matrix], input_length = tok_X_len, trainable = True))
            #   own inputs
#model.add(Embedding(round(vocab_size*1.5), 100, input_length = tok_X_len))          
            
# uncomment one of the above
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())
model.fit(tok_X_train_pad, np.array(sy_train), epochs = 5)#validation_split = 0.4)

loss, accuracy = model.evaluate(tok_X_test_pad, np.array(sy_test), verbose = 2)
print('Accuracy: %f' % (accuracy*100))

y_pred = model.predict_classes(tok_X_test_pad)

plot_roc_curve(sy_test, y_pred, 'CNN GLoVe and Own')

vocab = list(wv_mod.wv.vocab)
vocab_size = len(vocab)      

embedding_matrix = np.zeros((len(t.word_index.items())+1,70))        

for word,i in t.word_index.items():
    if word in vocab:
        embedding_matrix[i] = wv_mod[word]      # or wv_mod.wv.word_vec('good') --> for vector

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
print(nonzero_elements / len(t.word_index.items()))

t.fit_on_texts(sx_train)

tok_enc_x_train = t.texts_to_sequences(sx_train)
tok_enc_x_test = t.texts_to_sequences(sx_test)

vocab_size = len(t.word_index) + 1      # cg
max_length = max(max([len(i) for i in tok_enc_x_train]) , max([len(i) for i in tok_enc_x_test]))

tok_x_train = pad_sequences(tok_enc_x_train, maxlen = max_length, padding = 'post')     # Using 70 which was given to word2vec model instead of max_length
tok_x_test = pad_sequences(tok_enc_x_test, maxlen = max_length, padding = 'post')

print("\nWord2Vec Model\n\n")

model = Sequential()
model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix], input_length = max_length, trainable = True))
model.add(Conv1D(filters = 128, kernel_size = 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu',input_shape=(vocab_size,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(tok_x_train, sy_train, epochs = 5, verbose = 2, validation_split = 0.2)

loss, accuracy = model.evaluate(tok_x_test, sy_test, verbose=2)
print("Testing Accuracy:  {:.4f}".format(accuracy))

y_pred = model.predict_classes(tok_x_test)

plot_roc_curve(sy_test, y_pred, 'Word2Vec Model')


def plot_roc_curve(y_act, y_pred, name):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
        
    fpr, tpr, threshold = roc_curve(y_act, y_pred)
    auc_val = auc(fpr, tpr)
    
    print(name + ' AUC Value : ',auc_val)
    
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_val))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve'+name)
    plt.legend(loc='best')
    plt.show()
