import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def Visualization(path):
    # read data
    df = pd.read_table(path, sep=' ', header=None)
    data = np.array(df)
    data_list = data[0].astype(list)
    # print(data_list)
    epoch_list = [i+1 for i in range(len(data_list))]

    # Visualize
    plt.subplots(figsize=(10, 8))
    plt.plot(epoch_list,data_list,alpha=0.6,color='steelblue',label=path[:6],linewidth=5)
    plt.tick_params(labelsize=20)
    plt.xlabel('epochs', fontsize=20)
    plt.ylabel(path[:-4], fontsize=20)
    plt.legend(fontsize=15)
    plt.show()

if __name__ == '__main__' :
    Visualization('stage1_loss.txt')
    Visualization('stage2_loss.txt')