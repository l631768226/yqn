#!/Users/yqn/Anaconda/anaconda3/envs/test/bin/python3.7
import os
from PIL import Image  #PIL是python的第三方图像处理库
import shutil

# data_base_dir1:"/Users/yqn/Desktop/yqn可执行文件2/test/", 存放图片的路径
# write_file_name1:"/Users/yqn/Desktop/yqn可执行文件2/picture_test/文件名.txt"
# basedir:"/Users/yqn/Desktop/yqn可执行文件2/"
# dir_path:"/Users/yqn/Desktop/yqn可执行文件2/picture_test/images/"
# new_img_folder:"/Users/yqn/Desktop/yqn可执行文件2/picture_test/show/"
# openimgpath:"/Users/yqn/Desktop/yqn可执行文件2/val/"
# saveimg:"/Users/yqn/Desktop/yqn可执行文件2/picture_test/val1/"
def get_img(data_base_dir1,write_file_name1,basedir,dir_path,new_img_folder,openimgpath,saveimg):
    data_base_dir = data_base_dir1
    file_list = []  # 建立列表，用于保存图片信息
    # 读取图片文件，并将图片地址、图片名和标签写到txt文件中
    #write_file_name = '/Users/yqn/Desktop/yqn可执行文件/picture_test/文件名.txt'
    write_file_name = write_file_name1
    write_file = open(write_file_name, "w")  # 以只写方式打开write_file_name文件
    for file in os.listdir(data_base_dir):  # file为current_dir当前目录下图片名
        if file.endswith(".jpg"):  # 如果file以jpg结尾
            write_name = file  # 图片路径 + 图片名 + 标签
            file_list.append(write_name)  # 将write_name添加到file_list列表最后
    im = Image.open(openimgpath+'{}.jpg'.format(write_name[0:5]))
    im.save(saveimg+'{}.jpg'.format(write_name[0:5]))  # 把文件夹中指定的文件名称的图片另存到该路径下
    im.close()
    print(basedir)
    orderStr = "python " + basedir + "inTot.py --input_dir " + basedir + "picture_test/val1 --output_dir " + basedir + "picture_test --checkpoint "+ basedir + "train"
    print(orderStr)
    # os.system("python " + basedir + "inTot.py --input_dir " + basedir + "picture_test/val1 --output_dir " + basedir + "picture_test --checkpoint "+ basedir + "train")
    os.system(orderStr)

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.count('outputs') == 1:
                shutil.copy(os.path.join(root, file), new_img_folder)



def deleteall(path1):
    path=path1#path1指向picture_test文件
    #path = '/Users/yqn/Desktop/yqn可执行文件2/picture_test'
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)


# liucheng("/Users/yqn/Desktop/yqn可执行文件2/test/","/Users/yqn/Desktop/yqn可执行文件2/picture_test/文件名.txt"
#      ,"/Users/yqn/Desktop/yqn可执行文件2/","/Users/yqn/Desktop/yqn可执行文件2/picture_test/images/"
#     ,"/Users/yqn/Desktop/yqn可执行文件2/picture_test/show/","/Users/yqn/Desktop/yqn可执行文件2/val/",
#          "/Users/yqn/Desktop/yqn可执行文件2/picture_test/val1/")

# deleteall('/Users/yqn/Desktop/yqn可执行文件2/picture_test')

if __name__ == '__main__':
    get_img('E:/building/yqn/source/', 'E:/building/yqn/middle/tmp.txt', 'E:/building/yqn/',
             'E:/building/yqn/picture_test/images/', 'E:/building/yqn/picture_test/show/',
             'E:/building/yqn/val/', 'E:/building/yqn/picture_test/val1/')