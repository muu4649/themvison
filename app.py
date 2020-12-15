import os
from flask import Flask, request, redirect, url_for, make_response,render_template,send_file
from werkzeug.utils import secure_filename
from flask import send_from_directory
import matplotlib
matplotlib.use('Agg')
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm
from scipy.stats import gaussian_kde
from scipy.ndimage.filters import maximum_filter
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO


def detect_peaks(image, filter_size=3, order=0.15):##https://qiita.com/yoneda88/items/0cf4a9384c1c2203ea95
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image, mask=~(image == local_max))

    # 小さいピーク値を排除（最大ピーク値のorder倍以下のピークは排除）
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def func_search_neighbourhood(embs, p0,p01):#https://qiita.com/seigot/items/670032c12a38e64e20cb
    L = np.array([])
    for i in range(embs.shape[0]):
        norm = np.sqrt( (embs[i,0] - p0)*(embs[i,0] - p0) + (embs[i,1] - p01)*(embs[i,1] - p01) )
        L = np.append(L, norm)
    return np.argmin(L)
    #return L.argsort()[0], L.argsort()[1],L.argsort()[2]


# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['xlsx'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#@app.route('/uploads/<filename>')
## ファイルを表示する
#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def textminig(filename):
    data = pd.read_excel('./uploads/'+filename)
    x_data = np.array(list(data["word"]))
    assay = "IPC"
    le = LabelEncoder()#assay2num
    le = le.fit(data[assay])
    data[assay] = le.transform(data[assay])
    y_data = data[assay].values.reshape(-1,1)

    data1= pd.read_excel(r'./static/stopword.xlsx')
    stoplist = list(np.array(list(data1["Stop"])))

    #tf-idf
    vectorizer = TfidfVectorizer(max_df=0.7,min_df=0.0,use_idf=True, token_pattern=u'(?u)\\b\\w+\\b',stop_words=stoplist)
    vecs = vectorizer.fit_transform(x_data)

    #t_sne
    tsne = TSNE(random_state=0, n_iter=15000, metric='cosine')
    embs = tsne.fit_transform(vecs.toarray())

    #gaussian
    x = math.ceil(max((embs[:,0])))+1
    x1 = math.ceil(min((embs[:,0])))-1
    y = math.ceil(max((embs[:,1])))+1
    y1 = math.ceil(min((embs[:,1])))-1
    if x1<0:
        x1=-x1
    if y1<0:
        y1=-y1
    xx,yy = np.mgrid[-x1:x:1,-y1:y:1]
    positions = np.vstack([xx.ravel(),yy.ravel()])
    value = np.vstack([embs[:,0],embs[:,1]])

    kernel = gaussian_kde(value, bw_method='silverman')
    f = np.reshape(kernel(positions).T, xx.shape)
    maxid1=detect_peaks(f)


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.contourf(xx,yy,f, cmap=cm.jet)
    plt.scatter(embs[:, 0], embs[:, 1], c=y_data,s=3)


    words = vectorizer.get_feature_names()

    for i in range(len(maxid1[0])):
        plt.scatter(maxid1[0][i]-abs(x1), maxid1[1][i]-abs(y1), color='black',s=10)
        n1=func_search_neighbourhood(embs,maxid1[0][i]-abs(x1),maxid1[1][i]-abs(y1))

        w_id1=np.argsort(vecs[n1].toarray())[-1][-1]
        print(words[w_id1])
        w_id2=np.argsort(vecs[n1].toarray())[-1][-2]
        print(words[w_id2])
        
        plt.text(maxid1[0][i]-abs(x1), maxid1[1][i]-abs(y1), words[w_id1]+"\n"+words[w_id2])
   
    filename1="img.png"
    plt.savefig("./uploads/"+filename1)
    return filename1
    #return render_template('img.html', img_url=filename1)
    #return app.send_static_file('index.html')
    #return send_from_directory(app.config['UPLOAD_FOLDER'],filename1)

#def uploaded_file(filename):
#    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            print(file.filename)
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            print(filename)
            print(app.config['UPLOAD_FOLDER'])
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            fig_add = textminig(filename)
            # アップロード後のページに転送
           # filename1="img.png"
            return redirect(url_for('uploaded_file', filename=fig_add))
    return '''
    <html>
        <head>
            <meta charset="UTF-8">
            <title>
                ファイルをアップロードして俯瞰分析をしよう
            </title>
        </head>
        <body>
	    <center>
            <h1>
                ファイルをアップロードして俯瞰分析をしよう
            </h1>
            <img src="/static/sample.png" alt="sample">
            <form method = post enctype = multipart/form-data>
            <p><input type=file name = file>
            <input type = submit value = Upload>
            </form>
            <link rel="shortcut icon" href="{{ url_for('static', filename='favicon.ico') }}">
	    <center>
        </body> 
    '''
if __name__ == "__main__":
    app.run(debug=True)
