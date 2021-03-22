from flask import Flask
from flask import request
from liucheng import get_img
from lst_try import st_trajectory_model

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/liucheng', methods=['POST'])
def getImg():
    sourcePath = request.form.get("sourcePath")
    # print(sourcePath)
    txtPath = request.form.get("txtPath")
    # print(txtPath)
    basePath = request.form.get("basePath")
    # print(basePath)
    imgPath = request.form.get("imgPath")
    # print(imgPath)
    showPath = request.form.get("showPath")
    # print(showPath)
    valPath = request.form.get("valPath")
    # print(valPath)
    val1Path = request.form.get("val1Path")
    # print(val1Path)
    get_img(sourcePath, txtPath, basePath, imgPath, showPath, valPath, val1Path)
    return 'OK'
    # liucheng('E:/building/yqn/source/', 'E:/building/yqn/middle/tmp.txt', 'E:/building/yqn/',
    #          'E:/building/yqn/picture_test/images/', 'E:/building/yqn/picture_test/show/',
    #          'E:/building/yqn/val/', 'E:/building/yqn/picture_test/val1/')

@app.route('/path', methods=['POST'])
def get_path():
    path = request.form.get("path")
    return st_trajectory_model(path)

if __name__ == '__main__':
    app.run()
