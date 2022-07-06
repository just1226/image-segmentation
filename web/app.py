from flask import Flask,jsonify
from flask import request
from flask import render_template
from flask import send_file

import os
import io
import time
from tq_best import tq_test
from datetime import timedelta

ALLOWED_EXTENSIONS=set(['png','jpg','tif','JPG','PNG','TIF'])

app=Flask(__name__,static_folder='', static_url_path='')

# app.config['DEBUG'] =True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=5)

@app.route('/')
def Segmentation():
    return render_template('home.html')

#目标提取
@app.route('/objectTQ')
def objectTQ():
    return render_template('objectTQ.html')

@app.route('/save', methods=['GET', 'POST'])
def save():
        if request.method == 'POST' and request.files['img']:
            f = request.files['img']
            f.save('saved_files/tq/img.png')
            os.system('python ./tq_best/tq_test.py')
            num=1
            return render_template("result.html", data=num ,no_cache=time.time())

        else:
            return render_template('failure.html')

#目标检测
@app.route('/objectDetection')
def objectDetection():
    return render_template('objectDetection.html')

@app.route('/saveOD', methods=['GET', 'POST'])
def saveOD():
    if request.method == 'POST' and request.files['img']:
        f = request.files['img']
        f.save('saved_files/odt/img.png')
        os.system('python ./PPYOLO/main.py')
        num=2
        return render_template("result.html", data=num ,no_cache=time.time())

    else:
        return render_template('failure.html')



#变化检测
@app.route('/changeDetection')
def changeDetection():
    return render_template('changeDetection.html')

@app.route('/saveCD', methods=['GET', 'POST'])
def saveCD():
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']
        f1.save('saved_files/bit/A.png')
        f2.save('saved_files/bit/B.png')
        os.system('python ./BIT/test.py')
        num=3
        return render_template("result.html", data=num ,no_cache=time.time())

    else:
        return render_template('failure.html')

#地物分类
@app.route('/objectClass')
def objectClass():
    return render_template('objectClass.html')

@app.route('/saveCLS', methods=['GET', 'POST'])
def saveCLS():
    if request.method == 'POST' and request.files['img']:
        f = request.files['img']
        f.save('saved_files/cls/img.png')
        os.system('python ./classify/cls_test.py')
        num=4
        return render_template("result.html", data=num , no_cache=time.time())

    else:
        return render_template('failure.html')


if __name__=='__main__':
    app.run()