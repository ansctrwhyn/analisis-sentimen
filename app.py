from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from numpy.core.fromnumeric import mean
from sklearn import neighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, app, render_template, request, redirect, url_for, jsonify
import mysql.connector
import xlrd
import os
import pandas as pd
import string
import re
import time
import numpy as np
import tweepy
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
vectorizer = TfidfVectorizer()

app = Flask(__name__)

# Database
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sicepat"
)

mycursor = mydb.cursor(dictionary=True)

# Upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('home.html', menu="home")


@app.route('/dataset')
def show_dataset():
    return render_template('dataset.html', menu="dataset")


@app.route('/ajax_dataset', methods=["POST", "GET"])
def ajax_dataset():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM document")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM document limit %s, %s;", (row, rowperpage))
        documentlist = mycursor.fetchall()

        data = []
        for row in documentlist:
            data.append({
                'id': row['id_doc'],
                'dokumen': row['dokumen'],
                'sentimen': row['sentimen']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/dataset/importdata')
def import_data():
    return render_template('import_data.html', menu="dataset")


@app.route("/dataset/importdata", methods=['POST'])
def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], uploaded_file.filename)
       # set the file path
        uploaded_file.save(file_path)
        parseExcel(file_path)
       # save the file
    return redirect(url_for('show_dataset'))


def parseExcel(filePath):
    data = xlrd.open_workbook(filePath)
    sheet = data.sheet_by_index(0)
    for r in range(1, sheet.nrows):
        dokumen = sheet.cell(r, 0).value
        sentimen = sheet.cell(r, 1).value
        sql = "INSERT INTO document(dokumen, sentimen) VALUES (%s, %s)"
        values = (dokumen, sentimen)
        mycursor.execute(sql, values)
    mycursor.close()
    mydb.commit()
    mydb.close()


@app.route('/preprocessing')
def preprocessing():

    mycursor.execute("SELECT dokumen FROM document")
    documentlist = mycursor.fetchall()
    doc_list = [row['dokumen'] for row in documentlist]

    cleansing_result = [cleansing(text) for text in doc_list]
    for text in cleansing_result:
        mycursor.execute(
            "INSERT INTO cleansing(cleansing) VALUES(%s)", (text,))
    mydb.commit()

    casefolding_result = [case_folding(text) for text in cleansing_result]
    for text in casefolding_result:
        mycursor.execute(
            "INSERT INTO case_folding(case_folding) VALUES(%s)", (text,))
    mydb.commit()

    tokenizing_result = [tokenizing(text) for text in casefolding_result]
    for text in tokenizing_result:
        mycursor.execute(
            "INSERT INTO tokenizing(tokenizing) VALUES(%s)", (str(text),))
    mydb.commit()

    filtering_result = [filtering(text) for text in casefolding_result]
    for text in filtering_result:
        mycursor.execute(
            "INSERT INTO filtering(filtering) VALUES(%s)", (str(text),))
    mydb.commit()

    stemming_result = [stemming(text) for text in filtering_result]
    for text in stemming_result:
        mycursor.execute(
            "INSERT INTO stemming(stemming) VALUES(%s)", (str(text),))
    mydb.commit()

    return redirect(url_for('show_dataset'))


@app.route('/cleansing')
def show_cleansing():
    return render_template('cleansing.html', menu="preprocessing", submenu="cleansing")


@app.route('/ajax_cleansing', methods=["POST", "GET"])
def ajax_cleansing():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM cleansing")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM cleansing limit %s, %s;", (row, rowperpage))
        cleansinglist = mycursor.fetchall()

        data = []
        for row in cleansinglist:
            data.append({
                'id': row['id'],
                'cleansing': row['cleansing']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/casefolding')
def show_casefolding():
    return render_template('case_folding.html', menu="preprocessing", submenu="casefolding")


@app.route('/ajax_casefolding', methods=["POST", "GET"])
def ajax_casefolding():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM case_folding")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM case_folding limit %s, %s;", (row, rowperpage))
        cleansinglist = mycursor.fetchall()

        data = []
        for row in cleansinglist:
            data.append({
                'id': row['id'],
                'case_folding': row['case_folding']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/tokenizing')
def show_tokenizing():
    return render_template('tokenizing.html', menu="preprocessing", submenu="tokenizing")


@app.route('/ajax_tokenizing', methods=["POST", "GET"])
def ajax_tokenizing():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM tokenizing")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM tokenizing limit %s, %s;", (row, rowperpage))
        cleansinglist = mycursor.fetchall()

        data = []
        for row in cleansinglist:
            data.append({
                'id': row['id'],
                'tokenizing': row['tokenizing']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/filtering')
def show_filtering():
    return render_template('filtering.html', menu="preprocessing", submenu="filtering")


@app.route('/ajax_filtering', methods=["POST", "GET"])
def ajax_filtering():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM filtering")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM filtering limit %s, %s;", (row, rowperpage))
        cleansinglist = mycursor.fetchall()

        data = []
        for row in cleansinglist:
            data.append({
                'id': row['id'],
                'filtering': row['filtering']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/stemming')
def show_stemming():
    return render_template('stemming.html', menu="preprocessing", submenu="stemming")


@app.route('/ajax_stemming', methods=["POST", "GET"])
def ajax_stemming():
    if request.method == 'POST':
        draw = request.form['draw']
        row = int(request.form['start'])
        rowperpage = int(request.form['length'])
        print(draw)
        print(row)
        print(rowperpage)

        # Total number of records without filtering
        mycursor.execute("SELECT COUNT(*) AS allcount FROM stemming")
        rsallcount = mycursor.fetchone()
        totalRecords = rsallcount['allcount']
        print(totalRecords)

        mycursor.execute(
            "SELECT * FROM stemming limit %s, %s;", (row, rowperpage))
        cleansinglist = mycursor.fetchall()

        data = []
        for row in cleansinglist:
            data.append({
                'id': row['id'],
                'stemming': row['stemming']
            })

        response = {
            'draw': draw,
            'iTotalRecords': totalRecords,
            'iTotalDisplayRecords': 1000,
            'aaData': data,
        }
        return jsonify(response)


@app.route('/klasifikasi', methods=["POST", "GET"])
def form_klasifikasi():
    return render_template('klasifikasi.html', menu="klasifikasi")


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [cos_sim(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.flip(np.argsort(distances))[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


@app.route('/akurasi', methods=["POST", "GET"])
def akurasi():
    # mycursor.execute(
    #     "SELECT dokumen FROM document")
    # documentlist = mycursor.fetchall()
    # doc_list = [row['dokumen'] for row in documentlist]
    # cleansing_result = [cleansing(text) for text in doc_list]
    # casefolding_result = [case_folding(text) for text in cleansing_result]
    # tokenizing_result = [tokenizing(text) for text in casefolding_result]
    # filtering_result = [filtering(text) for text in casefolding_result]
    # stemming_result = [stemming(text) for text in filtering_result]
    # val = [['nama', 'saya', 'anisa'], ['cari', 'saya', 'disana']]

    if request.method == "POST":
        start = time.time()
        mycursor.execute('SELECT stemming from stemming')
        result = mycursor.fetchall()
        list1 = [row['stemming'] for row in result]
        new_strings = []

        for string in list1:
            new_string = eval(string)
            new_strings.append(new_string)
        mydb.commit()

        stemming_result = [' '.join(stemming(text)) for text in new_strings]
        tfidf = vectorizer.fit_transform(stemming_result)
        # tokens = vectorizer.get_feature_names()

        mycursor.execute(
            "SELECT sentimen FROM document")
        sentimenlist = mycursor.fetchall()
        sentimen = np.array([row['sentimen'] for row in sentimenlist])

        X = tfidf.toarray()
        y = sentimen

        kf = KFold(n_splits=10)
        akurasi = []
        precision = []
        recall = []
        f1score = []
        totalakurasi = 0
        totalprecision = 0
        totalrecall = 0
        totalf1score = 0
        k = 0
        n = int(request.form['neighbors'])
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = KNNClassifier(k=n)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            # print('hasil confusion matrix: \n', confusion_matrix(y_test, y_pred))
            nilaiakurasi = accuracy_score(y_test, y_pred)
            akurasi.append(nilaiakurasi)
            totalakurasi += nilaiakurasi
            nilaiprecision = precision_score(
                y_test, y_pred, average='weighted')
            precision.append(nilaiprecision)
            totalprecision += nilaiprecision
            nilairecall = recall_score(y_test, y_pred, average='weighted')
            recall.append(nilairecall)
            totalrecall += nilairecall
            nilaif1score = f1_score(y_test, y_pred, average='weighted')
            f1score.append(nilaif1score)
            totalf1score += nilaif1score
            k += 1
            # print('Akurasi: \n', accuracy_score(y_test, y_pred))
            # print(classification_report(y_test, y_pred))

        # print(akurasi)
        mean_accuracy = round(totalakurasi/k * 100, 2)
        mean_precision = round(totalprecision/k * 100, 2)
        mean_recall = round(totalrecall/k * 100, 2)
        mean_f1score = round(totalf1score/k * 100, 2)
        end = time.time() - start
        return render_template('akurasi.html', akurasi=mean_accuracy, precision=mean_precision, recall=mean_recall, f1score=mean_f1score, tetangga=n, time_taken=end, menu="akurasi")
    else:
        return render_template('klasifikasi.html', menu="klasifikasi")


def preprocessing(text):
    # doc_list = [row['dokumen'] for row in documentlist]
    cleansing_result = [cleansing(text) for text in text]
    casefolding_result = [case_folding(text) for text in cleansing_result]
    # tokenizing_result = [tokenizing(text) for text in casefolding_result]
    filtering_result = [filtering(text) for text in casefolding_result]
    stemming_result = [stemming(text) for text in filtering_result]
    return stemming_result


def cleansing(text):
    # hapus tab, newline, dan backslash
    text = text.replace('\t', ' ').replace('\n', ' ').replace('\\', ' ')
    # hapus underscore
    text = text.replace('_', '')
    # hapus user mention
    text = re.sub('@[A-Za-z0-9]+', '', text)
    # hapus link
    text = re.sub(
        '((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)+', '', text)
    # hapus hashtag
    text = re.sub('/#[\w_]+[ \t]*/', '', text)
    # hapus ASCII dan unicode
    text = text.encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    # hapus angka
    text = re.sub(r'\d+', '', text)
    # hapus punctuation
    text = text.translate(str.maketrans(
        string.punctuation, ' '*len(string.punctuation)))
    # hapus whitespace
    text = text.strip()
    # hapus multiple whitespace
#     text = ' '.join(text.split())
    text = re.sub('\s+', ' ', text)
    # hapus single character
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    return text


def case_folding(text):
    return text.lower()


def tokenizing(text):
    text = re.split('\W+', text)
    return text


def filtering(text):
    text = tokenizing(stopword.remove(text))
    return text


def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text


def tfidf(text):
    mycursor.execute(
        "SELECT dokumen FROM document")
    documentlist = mycursor.fetchall()
    doc_list = [row['dokumen'] for row in documentlist]
    cleansing_result = [cleansing(text) for text in doc_list]
    casefolding_result = [case_folding(text) for text in cleansing_result]
    # tokenizing_result = [tokenizing(text) for text in case_folding_result]
    filtering_result = [filtering(text) for text in casefolding_result]
    stemming_result = [stemming(text) for text in filtering_result]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(stemming_result)
    return(tfidf)


@app.route('/ujianalisis', methods=["POST", "GET"])
def uji_analisis():

    if request.method == "POST":
        n = int(request.form['count'])
        api = twitter_api()
        date = []
        text = []
        search_key = "sicepat_ekspres -filter:retweets"
        for tweet in tweepy.Cursor(api.search, q=search_key, lang='id', tweet_mode='extended', include_rts=False, exclude_replies=True).items(n):
            date.append(str(tweet.created_at))
            # query.append(tweet.text)
            try:
                text.append(tweet.retweeted_status.full_text)
            except AttributeError:  # Not a Retweet
                text.append(tweet.full_text)

        query = preprocessing(text)
        return render_template('uji_analisis.html', result=zip(date, query), menu="ujianalisis")
    else:

        return render_template('uji_analisis.html', menu="ujianalisis")


def pengujian():
    return


@app.route('/visualisasi', methods=["POST", "GET"])
def visualisasi():

    return render_template('visualisasi.html', menu="visualisasi")


def twitter_api():
    access_token = "1256453158601027584-B9Qk1xejLcwKTViW6cerDzUZUpaZuI"
    access_token_secret = "iN10UFyMoh41jFnky4UayQowYD3NUvXjc070R0QpyQ35P"
    api_key = "OGDqC3uu1NFSrJYx3vQNkzAKB"
    api_key_secret = "sy3BhmFn4gofZqswuHJ3CyW47JUO99UH6salTQyoCVdFkAtEdF"

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api


if __name__ == "__main__":
    app.run(debug=True)
