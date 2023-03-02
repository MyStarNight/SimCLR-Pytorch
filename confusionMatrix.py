from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

'''参考链接：http://t.csdn.cn/cVDN0'''

def confusion(y_pred,y_true):
    C = confusion_matrix(y_true,y_pred,labels=[0,1,2,3,4,5,6,7,8,9])
    plt.matshow(C,cmap=plt.cm.Reds)

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()