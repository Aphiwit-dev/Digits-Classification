from sklearn import svm
from sklearn import datasets
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
n_samples = len(digits.images)
Xtrain = digits.images.reshape(n_samples,-1)
ytrain = digits.target
model = svm.SVC(gamma = 'scale')
model.fit(Xtrain,ytrain)

def predict_digit(imgfiles):
    img = Image.open(imgfiles).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((8,8),Image.ANTIALIAS)
    pixel = np.array(img)
    pixel = pixel.astype('int')

    my_digit = pixel.reshape(1,-1)
    predicted = model.predict(my_digit)

    plt.figure(figsize=(2,2))
    plt.imshow(pixel,cmap = plt.cm.gray_r)
    plt.title('predicted : {}'.format(predicted[0]))
    plt.xticks()
    plt.yticks()
    plt.show()

imgfiles = ['digit_input1.jpg','digit_input2.jpg','digit_input4.jpg','digit_input7.jpg'] #ตัวอย่างการใส่ชื่อไฟล์ภาพที่ต้องการทำนาย
for i in imgfiles:
    predict_digit(i)
    