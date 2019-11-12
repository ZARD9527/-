from django.shortcuts import render, render_to_response
import os
from django.http import HttpResponse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from django.views.decorators.csrf import csrf_exempt

from sklearn.externals import joblib
# Create your views here.

def index(req):
    return render_to_response('index.html')

def about_nursery(req):
    return render_to_response('about_nursery.html')

def about_titanic(req):
    return render_to_response('about_titanic.html')

def about_credit(req):
    return render_to_response('about_credit.html')

def about_breast(req):
    return render_to_response('about_breast.html')

def data(req):
    return render_to_response('portfolio.html')

def portfolio(request):
    return render_to_response('portfolio.html')

# 下面代码是新加的测试代码2018.1.26 功能是上传文件的
def hello(request):
    if request.method == 'GET':
        return render(request, 'portfolio.html')
    elif request.method == 'POST':
        obj = request.FILES.get('f')
        f = open(os.path.join('static/uploadfile', obj.name), 'wb')
        for line in obj.chunks():
            f.write(line)
        f.close()
        return render_to_response('portfolio.html', {"result": "上传成功"})

def TreeModel(req):
    path = "static/data/nursery.csv"
    data = pd.read_csv(path, header=None)
    data.columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data['label'].values))
    data['label'] = lbl.transform(list(data['label'].values))
    # print(data["label"][:5])
    lbl.fit(list(data['parents'].values))
    data['parents'] = lbl.transform(list(data['parents'].values))
    lbl.fit(list(data['has_nurs'].values))
    data['has_nurs'] = lbl.transform(list(data['has_nurs'].values))
    lbl.fit(list(data['form'].values))
    data['form'] = lbl.transform(list(data['form'].values))
    lbl.fit(list(data['children'].values))
    data['children'] = lbl.transform(list(data['children'].values))
    lbl.fit(list(data['housing'].values))
    data['housing'] = lbl.transform(list(data['housing'].values))
    lbl.fit(list(data['finance'].values))
    data['finance'] = lbl.transform(list(data['finance'].values))
    lbl.fit(list(data['social'].values))
    data['social'] = lbl.transform(list(data['social'].values))
    lbl.fit(list(data['health'].values))
    data['health'] = lbl.transform(list(data['health'].values))
    clf = tree.DecisionTreeClassifier(max_depth=8)
    y_label = data["label"].values
    x_prediction = data[["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]]
    X_train, X_test, y_train, y_test = train_test_split(x_prediction, y_label, random_state=0)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return render_to_response('portfolio.html', {"data": np.mean(y_test == y_pred)})

def docs(req):
    return render_to_response('docs.html')

# checkbox demo
def check_box(request):
    if request.method == "POST":
        check_box_list = request.POST.getlist('vehicle')
        if check_box_list:
            print(check_box_list)
            return HttpResponse("ok")
        else:
            print("fail")
            return HttpResponse("fail")
    else:
        a=[1, 2, 3, 4]
        return render_to_response('demo.html', {'a': a})

def add(request):
    a = request.GET['a']
    b = request.GET['b']
    a = int(a)
    b = int(b)
    return render_to_response({'sum': a+b})

def xxx(request):
    return render_to_response('demo.html')

def upload(request):
    if request.method == 'GET':
        return render(request, 'predict.html')
    elif request.method == 'POST':
        file = request.FILES.get('f')
        f = open(os.path.join('static/uploadfile', file.name), 'wb')
        for line in file.chunks():
            f.write(line)
        f.close()
        return render_to_response("predict.html", {"result": "上传成功"}) # 跳转页面

def model_select(request):
    if request.method == "GET":
        return render(request, "about_nursery.html")
    elif request.method == 'POST':
        file_ = request.FILES.get('f')
        f = open(os.path.join('static/uploadfile', file_.name), 'wb')
        for line in file_.chunks():
            f.write(line)
        f.close()

        check_list = request.POST.getlist('checkbox_1')
        path = "static/uploadfile/"+file_.name
        data = pd.read_csv(path, header=None)
        data.columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data['label'].values))
        data['label'] = lbl.transform(list(data['label'].values))
        # print(data["label"][:5])
        lbl.fit(list(data['parents'].values))
        data['parents'] = lbl.transform(list(data['parents'].values))
        lbl.fit(list(data['has_nurs'].values))
        data['has_nurs'] = lbl.transform(list(data['has_nurs'].values))
        lbl.fit(list(data['form'].values))
        data['form'] = lbl.transform(list(data['form'].values))
        lbl.fit(list(data['children'].values))
        data['children'] = lbl.transform(list(data['children'].values))
        lbl.fit(list(data['housing'].values))
        data['housing'] = lbl.transform(list(data['housing'].values))
        lbl.fit(list(data['finance'].values))
        data['finance'] = lbl.transform(list(data['finance'].values))
        lbl.fit(list(data['social'].values))
        data['social'] = lbl.transform(list(data['social'].values))
        lbl.fit(list(data['health'].values))
        data['health'] = lbl.transform(list(data['health'].values))
        y_label = data["label"].values
        x_prediction = data[["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]]
        clf = joblib.load("static/trainmodel/nursery_model.m")
        result_pred = clf.predict(x_prediction)
        label = {"label"}
        test = pd.DataFrame(columns=label, data=result_pred)
        test.to_csv('static/predict_result/'+"result"+file_.name)
        result_file = '/static/predict_result/'+"result"+file_.name
        return render_to_response("about_nursery.html", {"check_list___": result_file})#checkbox.html


def deal_data():
    path = "static/data/nursery.csv"
    data = pd.read_csv(path, header=None)
    data.columns = ["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health", "label"]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(data['label'].values))
    data['label'] = lbl.transform(list(data['label'].values))
    # print(data["label"][:5])
    lbl.fit(list(data['parents'].values))
    data['parents'] = lbl.transform(list(data['parents'].values))
    lbl.fit(list(data['has_nurs'].values))
    data['has_nurs'] = lbl.transform(list(data['has_nurs'].values))
    lbl.fit(list(data['form'].values))
    data['form'] = lbl.transform(list(data['form'].values))
    lbl.fit(list(data['children'].values))
    data['children'] = lbl.transform(list(data['children'].values))
    lbl.fit(list(data['housing'].values))
    data['housing'] = lbl.transform(list(data['housing'].values))
    lbl.fit(list(data['finance'].values))
    data['finance'] = lbl.transform(list(data['finance'].values))
    lbl.fit(list(data['social'].values))
    data['social'] = lbl.transform(list(data['social'].values))
    lbl.fit(list(data['health'].values))
    data['health'] = lbl.transform(list(data['health'].values))
    y_label = data["label"].values
    x_prediction = data[["parents", "has_nurs", "form", "children", "housing", "finance", "social", "health"]]
    return y_label,x_prediction

def  decsionTree_model():
    text_data = deal_data()[1]
    clf = joblib.load("static/trainmodel/nursery.m")
    result_pred = clf.predict(text_data)
    label = {"label"}
    test = pd.DataFrame(columns=label, data=result_pred)
    return test.to_csv('static/predict_result/xxxx.csv')


def clf_model(request):
    if request.method =="POST":
        check_list = request.POST.getlist('checkbox_1')
        if "DecsionTree" in check_list:
            return HttpResponse(decsionTree_model())

        # if "RandomForest" in check_list:
        #
        # if "AdaBoost" in check_list:
        #
        # if "ExtraTrees" in check_list: