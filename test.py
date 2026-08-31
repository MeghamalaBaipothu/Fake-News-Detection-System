import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from FireflyMSVM import FireflyMSVM
from sklearn import svm
import os
import pickle
'''
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
'''
#define object to remove stop words and other text processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

dataset = pd.read_csv("Dataset/politifact.csv")
labels = dataset['target'].ravel()
news = dataset['News'].ravel()

#define function to clean text by removing stop words and other special symbols
def cleanText(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation) #remove punctuation
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]#take only alphabets
    tokens = [w for w in tokens if not w in stop_words]#remove stop words
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [ps.stem(token) for token in tokens] #apply stemming and lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens
'''
X = []
Y = []
for i in range(len(news)):
    data = str(news[i]).strip()
    data = data.lower()
    if len(data) > 0:
        data = cleanText(data)
        label = 0
        if labels[i] == "TRUE":
            label = 1
        X.append(data)
        Y.append(label)
        print(str(i)+" "+str(len(data))+" "+str(label))

X = np.asarray(X)
Y = np.asarray(Y)
np.save("model/X", X)
np.save("model/Y", Y)
'''
X = np.load("model/X.npy")
Y = np.load("model/Y.npy")

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)
X = tfidf_vectorizer.fit_transform(X).toarray()
print(X.shape)
print(Y.shape)
print(np.unique(Y, return_counts=True))

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=300)
'''
f = open('model/pca.pckl', 'wb')
pickle.dump(pca, f)
f.close()
'''
f = open('model/pca.pckl', 'rb')
pca = pickle.load(f)
f.close()
X = pca.transform(X)
print(X.shape)

if os.path.exists("model/firefly.npy"):
    selected_features = np.load("model/firefly.npy")
    X = X[:, selected_features]
else:
    firefly_msvm = FireflyMSVM(n_fireflies=5, max_iterations=1)
    X, selected_features = firefly_msvm.fit_transform(X, Y)
    np.save("model/firefly", selected_features)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

svm_cls = svm.SVC(C=400)
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
acc = accuracy_score(y_test, predict)
print(acc)

data = pd.read_csv('Dataset/testNews.csv',encoding = "ISO-8859-1")
news = data.values
data = data.values
temp = []
for i in range(len(data)):
    value = data[i,0]
    value = value.strip().lower()
    value = cleanText(value)
    temp.append(value)
data = tfidf_vectorizer.transform(temp).toarray()
data = scaler.transform(data)
data = pca.transform(data)
data = data[:, selected_features]
predict = svm_cls.predict(data)
print(predict)
'''
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
X_train1 = np.reshape(X_train, (X_train.shape[0], 16, 10))
X_test1 = np.reshape(X_test, (X_test.shape[0], 16, 10))
lstm_model = Sequential()#defining deep learning sequential object
#adding LSTM layer with 100 filters to filter given input X train data to select relevant features
lstm_model.add(LSTM(32,input_shape=(X_train1.shape[1], X_train1.shape[2])))
#adding dropout layer to remove irrelevant features
lstm_model.add(Dropout(0.3))
#adding another layer
lstm_model.add(Dense(32, activation='relu'))
#defining output layer for prediction
lstm_model.add(Dense(y_train1.shape[1], activation='softmax'))
#compile LSTM model
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#start training model on train data and perform validation on test data
#train and load the model
if os.path.exists("model/lstm_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/lstm_weights.hdf5', verbose = 1, save_best_only = True)
    hist = lstm_model.fit(X_train1, y_train1, batch_size = 32, epochs = 35, validation_data=(X_test1, y_test1), callbacks=[model_check_point], verbose=1)
    f = open('model/lstm_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    lstm_model.load_weights("model/lstm_weights.hdf5")
#perform prediction on test data    
predict = lstm_model.predict(X_test1)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test1, axis=1)
acc = accuracy_score(y_test1, predict)
print(acc)
'''






        
