import random
import os
from PIL import Image

def train_test_split(filename, destination):
    
    filesource = "../" + filename + "/"
    dst = "../" + destination + "/test/"
    path, dirs, files = next(os.walk(filesource))

    for dir in dirs:
        src = (filesource + dir)
        path2, dirs2, files2 = next(os.walk(src))  
        random.seed(7) #arbitrarily chosen to repeat pseudo-randomness
        images_moving = random.sample(files2, round(0.2 * len(files2)))
        for img in images_moving:
            fullPath = src + "/" + img
            os.system("mv" + " " + fullPath + " " + dst + dirs2 + "/")


#Not Necessary with flow_from_directory in Keras
def resize_images(width, height, filename):

    filesource = "../" + filename + "/"
    extensions = '.jpg'
    path, dirs, files = next(os.walk(filesource))
    
    for dir in dirs:
        subpath, subdirs, subfiles = next(os.walk(filesource + dir))
        for img in subfiles:
            path = filesource + dir + "/" + img
            imgname, img_ext = os.path.splitext(path)
            if img_ext == extensions:
                image = Image.open(path)
                image = image.resize((width, height), PIL.Image.ANTIALIAS)
                image.save(img)
        for subdir in subdirs:
            subpath2, subdirs2, subfiles2 = next(os.walk(filesource + dir
                                                         + "/" + subdir))
            for img in subfiles2:
                path = filesource + dir + "/" + subdir + "/" + img
                imgname, img_ext = os.path.splitext(path)
                if img_ext == extensions:
                    image = Image.open(path)
                    image = image.resize((width, height), PIL.Image.ANTIALIAS)
                    image.save(img)