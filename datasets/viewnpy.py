"""
  @Author  : Xinyang Li
  @Time    : 2022/11/19 上午2:16
"""
import os.path

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    hm_0 = np.load(os.path.join('./sample/B.assignExp/heatmap/20200821152001999_7824_824122_LVT04__0___0____hm____0.npy'))
    hm_1 = np.load(
        os.path.join('./sample/B.assignExp/heatmap/20200821152001999_7824_824122_LVT04__0___0____hm____1.npy'))

    plt.figure()
    plt.imshow(hm_0, cmap='jet')
    plt.show()

    plt.figure()
    plt.imshow(hm_1, cmap='jet')
    plt.show()

    # off = np.load(
    #     os.path.join('./sample/B.assignExp/offset/20200821152001999_7824_824122_LVT04__512___256____off____1.npy'))

    print('done')