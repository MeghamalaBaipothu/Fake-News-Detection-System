from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import pymysql
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
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
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import seaborn as sns
from sklearn.metrics import confusion_matrix

global username, X, Y, scaler, pca, svm_cls, selected_features, tfidf_vectorizer
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = [] 
fscore = []
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

if os.path.exists("model/X.npy"):
    X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
else:
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
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save("model/X", X)
    np.save("model/Y", Y)

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def FeaturesSelection(request):
    if request.method == 'GET':
        global X, Y, scaler, pca, svm_cls, selected_features, tfidf_vectorizer
        global X_train, X_test, y_train, y_test
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words)
        X = tfidf_vectorizer.fit_transform(X).toarray()
        output = "Total records found in Dataset = "+str(X.shape[0])+"<br/>"
        output += "Total features found in Dataset = "+str(X.shape[1])+"<br/>"
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca = PCA(n_components=300)
        f = open('model/pca.pckl', 'rb')
        pca = pickle.load(f)
        f.close()
        X = pca.transform(X)
        output += "Total features extracted using MPCA = "+str(X.shape[1])+"<br/>"
        if os.path.exists("model/firefly.npy"):
            selected_features = np.load("model/firefly.npy")
            X = X[:, selected_features]
        else:
            firefly_msvm = FireflyMSVM(n_fireflies=5, max_iterations=1)
            X, selected_features = firefly_msvm.fit_transform(X, Y)
            np.save("model/firefly", selected_features)
        output += "Total features selected using Firefly-MSVM Optimization = "+str(X.shape[1])+"<br/><br/>"
        output += "Dataset Train & Test Split Details"
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        output += "80% dataset records using to train Algorithms = "+str(X_train.shape[0])+"<br/>"
        output += "20% dataset records using to test Algorithms = "+str(X_test.shape[0])+"<br/>"
        data = np.load("model/data.npy", allow_pickle=True)
        X_train, X_test, y_train, y_test = data
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def PredictFileAction(request):
    if request.method == 'POST':
        myfile = request.FILES['t2'].read()
        if os.path.exists('FakeApp/static/test.csv'):
            os.remove('FakeApp/static/test.csv')
        with open('FakeApp/static/test.csv', "wb") as file:
            file.write(myfile)
        file.close()
        data = pd.read_csv('FakeApp/static/test.csv',encoding = "ISO-8859-1")
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
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Test News</th><th><font size="" color="black">Prediction Status</th>'
        output+='</tr>'
        for i in range(len(predict)):
            if predict[i] == 0:
                output += '<tr><td>'+str(news[i])+"</td><td><font size=4 color=red>Fake</font></td></tr>"
            else:
                output += '<tr><td>'+str(news[i])+"</td><td><font size=4 color=green>True</font></td></tr>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)        

def PredictAction(request):
    if request.method == 'POST':
        global scaler, pca, svm_cls, selected_features, tfidf_vectorizer, labels
        news = request.POST.get('t1', False)
        data = news.strip().lower()
        data = cleanText(data)
        data = tfidf_vectorizer.transform([data]).toarray()
        data = scaler.transform(data)
        data = pca.transform(data)
        data = data[:, selected_features]
        predict = svm_cls.predict(data)[0]
        output = "<font size=3 color=green>Given news predicted as TRUE</font>"
        if predict == 0:
            output = "<font size=3 color=red>Given news predicted as FAKE</font>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RunML(request):
    if request.method == 'GET':
        global X, Y, scaler, pca, svm_cls, selected_features, tfidf_vectorizer
        global X_train, X_test, y_train, y_test
        global accuracy, precision, recall, fscore
        class_label = ['False', 'True']
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        svm_cls = svm.SVC(C=400)
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)
        calculateMetrics("Propose MSVM", y_test, predict)
        print(accuracy)
        conf_matrix = confusion_matrix(y_test, predict)
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
        calculateMetrics("Existing LSTM", y_test, predict)
        print(accuracy)
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Propose MSVM', 'Existing LSTM']
        for i in range(len(algorithms)):
            output += '<td><font size="" color="black">'+algorithms[i]+'</td><td><font size="" color="black">'+str(accuracy[i])+'</td><td><font size="" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="" color="black">'+str(recall[i])+'</td><td><font size="" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        df = pd.DataFrame([['Propose MSVM','Accuracy',accuracy[0]],['Propose MSVM','Precision',precision[0]],['Propose MSVM','Recall',recall[0]],['Propose MSVM','FSCORE',fscore[0]],
                           ['LSTM','Accuracy',accuracy[1]],['LSTM','Precision',precision[1]],['LSTM','Recall',recall[1]],['LSTM','FSCORE',fscore[1]],
                          ],columns=['Parameters','Algorithms','Value'])

        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = class_label, yticklabels = class_label, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(class_label)])    
        df.pivot("Parameters", "Algorithms", "Value").plot(ax=axis[1], kind='bar')
        plt.title("All Algorithms Performance Graph")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global dataset
        columns = dataset.columns
        data = dataset.values
        output='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output += '<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(data)):
            output += '<tr>'
            for j in range(len(data[i])):
                output += '<td><font size="3" color="black">'+str(data[i,j])+'</td>'
            output += '</tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = os.getenv('DB_PASSWORD'), database = 'fakenews',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
               
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = os.getenv('DB_PASSWORD'), database = 'fakenews',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = os.getenv('DB_PASSWORD'), database = 'fakenews',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform fake news prediction"
        context= {'data':output}
        return render(request, 'Register.html', context)       

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})



    

