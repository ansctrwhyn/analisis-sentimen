from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
# from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from flask import Flask, app, render_template, request, redirect, url_for, jsonify, session, flash
from flask_session import Session
import mysql.connector
import xlrd
import os
import string
import re
import time
import numpy as np
import tweepy
import datetime
import pytz

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
vectorizer = TfidfVectorizer()

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

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
        i = 1
        for row in documentlist:
            data.append({
                'id': i,
                'dokumen': row['dokumen'],
                'sentimen': row['sentimen']
            })
            i += 1

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
        i = 1
        for row in cleansinglist:
            data.append({
                'id': i,
                'cleansing': row['cleansing']
            })
            i += 1

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
        casefoldinglist = mycursor.fetchall()

        data = []
        i = 1
        for row in casefoldinglist:
            data.append({
                'id': i,
                'case_folding': row['case_folding']
            })
            i += 1

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
        tokenizinglist = mycursor.fetchall()

        data = []
        i = 1
        for row in tokenizinglist:
            data.append({
                'id': i,
                'tokenizing': row['tokenizing']
            })
            i += 1

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
        filteringlist = mycursor.fetchall()

        data = []
        i = 1
        for row in filteringlist:
            data.append({
                'id': i,
                'filtering': row['filtering']
            })
            i += 1

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
        stemminglist = mycursor.fetchall()

        data = []
        i = 1
        for row in stemminglist:
            data.append({
                'id': i,
                'stemming': row['stemming']
            })
            i += 1

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


class KNNCosine:
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


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))


class KNNEuclidean:
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
        distances = [euclidean_distance(x, x_train)
                     for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]


@app.route('/akurasi', methods=["POST", "GET"])
def akurasi():

    if request.method == "POST":

        # start = time.time()
        mycursor.execute('SELECT stemming from stemming')
        result = mycursor.fetchall()
        list1 = [row['stemming'] for row in result]
        new_strings = []

        for string in list1:
            new_string = eval(string)
            new_strings.append(new_string)
        mydb.commit()

        stemming_result = [' '.join(text) for text in new_strings]
        tfidf = vectorizer.fit_transform(stemming_result)
        # tokens = vectorizer.get_feature_names()

        mycursor.execute(
            "SELECT sentimen FROM document")
        sentimenlist = mycursor.fetchall()
        sentimen = np.array([row['sentimen'] for row in sentimenlist])

        X = tfidf.toarray()
        y = sentimen

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
        distance = request.form['distance']
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if distance == 'cosine':
                knn = KNNCosine(k=n)
            else:
                knn = KNNEuclidean(k=n)

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
        # end = time.time() - start
        return render_template('akurasi.html', akurasi=mean_accuracy, precision=mean_precision, recall=mean_recall, f1score=mean_f1score, tetangga=n, menu="akurasi")
    else:
        flash("Klasifikasi")
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
    text = ' '.join(word for word in text.split() if len(word) > 1)
    # hapus kata yang mengandung multiple same character
    # text = re.compile(r'(.)\1{1,}', re.IGNORECASE).sub(r'\1', text)
    # replace repeated character to single character
    text = re.sub("(.)\\1{2,}", "\\1", text)
    # hapus single character
    text = ' '.join(word for word in text.split() if len(word) > 1)
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


@app.route('/pengujian1', methods=["POST", "GET"])
def pengujian1():

    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    if not session.get('a1') is None:
        # session.clear()
        return render_template('pengujian1.html', result=zip(k_values, session['a1'], session['p1'], session['r1'], session['f1'], session['t1']), menu="pengujian", submenu="pengujian1")
    else:
        mycursor.execute('SELECT stemming from stemming')
        result = mycursor.fetchall()
        list1 = [row['stemming'] for row in result]
        new_strings = []

        for string in list1:
            new_string = eval(string)
            new_strings.append(new_string)
        mydb.commit()

        stemming_result = [' '.join(text) for text in new_strings]
        tfidf = vectorizer.fit_transform(stemming_result)
        # tokens = vectorizer.get_feature_names()

        mycursor.execute(
            "SELECT sentimen FROM document")
        sentimenlist = mycursor.fetchall()
        sentimen = np.array([row['sentimen'] for row in sentimenlist])

        X = tfidf.toarray()
        # X = np.array(stemming_result)
        y = sentimen

        a = []
        p = []
        r = []
        f = []
        t = []

        for k in k_values:
            start = time.time()
            akurasi = []
            precision = []
            recall = []
            f1score = []
            totalakurasi = 0
            totalprecision = 0
            totalrecall = 0
            totalf1score = 0
            cv = 0
            # n = int(request.form['neighbors'])
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # X_train = vectorizer.fit_transform(X_train)
                # X_test = vectorizer.transform(X_test)

                knn = KNNCosine(k=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                nilaiakurasi = accuracy_score(y_test, y_pred)
                akurasi.append(nilaiakurasi)
                totalakurasi += nilaiakurasi
                nilaiprecision = precision_score(
                    y_test, y_pred, average='macro')
                precision.append(nilaiprecision)
                totalprecision += nilaiprecision
                nilairecall = recall_score(y_test, y_pred, average='macro')
                recall.append(nilairecall)
                totalrecall += nilairecall
                nilaif1score = f1_score(y_test, y_pred, average='macro')
                f1score.append(nilaif1score)
                totalf1score += nilaif1score
                cv += 1

            mean_accuracy = round(totalakurasi/cv * 100, 2)
            mean_precision = round(totalprecision/cv * 100, 2)
            mean_recall = round(totalrecall/cv * 100, 2)
            mean_f1score = round(totalf1score/cv * 100, 2)
            a.append(mean_accuracy)
            p.append(mean_precision)
            r.append(mean_recall)
            f.append(mean_f1score)
            end = round(time.time() - start, 2)
            t.append(end)

        session['a1'] = a
        session['p1'] = p
        session['r1'] = r
        session['f1'] = f
        session['t1'] = t
        return render_template('pengujian1.html', result=zip(k_values, session['a1'], session['p1'], session['r1'], session['f1'], session['t1']), menu="pengujian", submenu="pengujian1")


@app.route('/pengujian2', methods=["POST", "GET"])
def pengujian2():
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

    if not session.get('a2') is None:
        return render_template('pengujian2.html', result=zip(k_values, session['a2'], session['p2'], session['r2'], session['f2'], session['t2']), menu="pengujian", submenu="pengujian2")
    else:
        mycursor.execute('SELECT stemming from stemming')
        result = mycursor.fetchall()
        list1 = [row['stemming'] for row in result]
        new_strings = []

        for string in list1:
            new_string = eval(string)
            new_strings.append(new_string)
        mydb.commit()

        stemming_result = [' '.join(text) for text in new_strings]
        tfidf = vectorizer.fit_transform(stemming_result)
        # tokens = vectorizer.get_feature_names()

        mycursor.execute(
            "SELECT sentimen FROM document")
        sentimenlist = mycursor.fetchall()
        sentimen = np.array([row['sentimen'] for row in sentimenlist])

        X = tfidf.toarray()
        y = sentimen

        a = []
        p = []
        r = []
        f = []
        t = []

        for k in k_values:
            start = time.time()
            akurasi = []
            precision = []
            recall = []
            f1score = []
            totalakurasi = 0
            totalprecision = 0
            totalrecall = 0
            totalf1score = 0
            cv = 0
            # n = int(request.form['neighbors'])
            kf = KFold(n_splits=10)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                knn = KNNEuclidean(k=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                nilaiakurasi = accuracy_score(y_test, y_pred)
                akurasi.append(nilaiakurasi)
                totalakurasi += nilaiakurasi
                nilaiprecision = precision_score(
                    y_test, y_pred, average='macro')
                precision.append(nilaiprecision)
                totalprecision += nilaiprecision
                nilairecall = recall_score(y_test, y_pred, average='macro')
                recall.append(nilairecall)
                totalrecall += nilairecall
                nilaif1score = f1_score(y_test, y_pred, average='macro')
                f1score.append(nilaif1score)
                totalf1score += nilaif1score
                cv += 1

            mean_accuracy = round(totalakurasi/cv * 100, 2)
            mean_precision = round(totalprecision/cv * 100, 2)
            mean_recall = round(totalrecall/cv * 100, 2)
            mean_f1score = round(totalf1score/cv * 100, 2)
            a.append(mean_accuracy)
            p.append(mean_precision)
            r.append(mean_recall)
            f.append(mean_f1score)
            end = round(time.time() - start, 2)
            t.append(end)

        session['a2'] = a
        session['p2'] = p
        session['r2'] = r
        session['f2'] = f
        session['t2'] = t
        return render_template('pengujian2.html', result=zip(k_values, session['a2'], session['p2'], session['r2'], session['f2'], session['t2']), menu="pengujian", submenu="pengujian2")


def convert_datetime_timezone(dt, tz1, tz2):
    tz1 = pytz.timezone(tz1)
    tz2 = pytz.timezone(tz2)

    dt = datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    dt = tz1.localize(dt)
    dt = dt.astimezone(tz2)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")

    return dt


@app.route('/pengujian3', methods=["POST", "GET"])
def pengujian3():

    if request.method == "POST":
        if 'count' in request.form:
            date = []
            text = []
            n = int(request.form['count'])
            api = twitter_api()
            search_key = "sicepat -from:sicepat_ekspres -filter:retweets"
            for tweet in tweepy.Cursor(api.search, q=search_key, lang='id', result_type="recent", tweet_mode='extended', include_rts=False, exclude_replies=True).items(n):
                utc = str(tweet.created_at)
                local = convert_datetime_timezone(utc, "UTC", "Asia/Jakarta")
                date.append(local)
                # query.append(tweet.text)

                try:
                    text.append(tweet.retweeted_status.full_text)
                except AttributeError:  # Not a Retweet
                    text.append(tweet.full_text)

            session['date'] = date
            session['text'] = text
            prediksi = []
            for _ in range(len(session['text'])):
                prediksi.append('Belum diketahui')

            return render_template('pengujian3.html', result=zip(session['date'], session['text'], prediksi), menu="pengujian", submenu="pengujian3")

        if 'neighbors' in request.form:
            mycursor.execute('SELECT stemming from stemming')
            result = mycursor.fetchall()
            list1 = [row['stemming'] for row in result]
            new_strings = []

            for string in list1:
                new_string = eval(string)
                new_strings.append(new_string)
            mydb.commit()

            stemming_result = [' '.join(text) for text in new_strings]
            tfidf = vectorizer.fit_transform(stemming_result)

            mycursor.execute(
                "SELECT sentimen FROM document")
            sentimenlist = mycursor.fetchall()
            sentimen = np.array([row['sentimen'] for row in sentimenlist])

            X = tfidf.toarray()
            y = sentimen

            vectorizer_uji = TfidfVectorizer(
                vocabulary=vectorizer.vocabulary_)
            text_preprocessing = preprocessing(session['text'])
            text = [' '.join(text) for text in text_preprocessing]
            tfidf_query = vectorizer_uji.fit_transform(text)
            query = tfidf_query.toarray()

            n = int(request.form['neighbors'])
            distance = request.form['distance']
            if distance == 'cosine':
                knn = KNNCosine(k=n)
            else:
                knn = KNNEuclidean(k=n)
            knn.fit(X, y)
            query_pred = knn.predict(query)
            session['prediksi'] = query_pred
            return render_template('pengujian3.html', result=zip(session['date'], session['text'], query_pred), menu="pengujian", submenu="pengujian3")
        # session.pop('date')
        # session.pop('text')
    else:
        if not session.get('prediksi') is None:
            return render_template('pengujian3.html', result=zip(session['date'], session['text'], session['prediksi']), menu="pengujian", submenu="pengujian3")
        else:
            return render_template('pengujian3.html', menu="pengujian", submenu="pengujian3")


@app.route('/visualisasi', methods=["POST", "GET"])
def visualisasi():
    if not session.get('prediksi') is None:
        list_positif = []
        list_negatif = []
        for data in session['prediksi']:
            if data == 'Positif':
                list_positif.append(data)
            else:
                list_negatif.append(data)

        count_positif = len(list_positif)
        count_negatif = len(list_negatif)
        positif = round(len(list_positif) /
                        (len(session['prediksi'])) * 100, 2)
        negatif = round(len(list_negatif) /
                        (len(session['prediksi'])) * 100, 2)

        return render_template('visualisasi.html', count_positif=count_positif, count_negatif=count_negatif, positif=positif, negatif=negatif, menu="visualisasi")
    else:
        flash("Uji Analisis Sentimen")
        return render_template('uji_analisis.html', menu="ujianalisis")


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
