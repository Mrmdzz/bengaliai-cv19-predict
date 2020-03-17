
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import numpy as np # linear algebra
import pandas as pd 
from tqdm.auto import tqdm
from glob import glob
import time, gc
import cv2
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Dropout
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont
import tensorflow.keras as tfkeras
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow_core.python.keras import backend 
#from tensorflow.keras import initializers
#from tensorflow.python.keras.engine.base_layer import InputSpec, Layer
test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')
sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
print(f'Size of test data: {test_df_.shape}')
Listname=['consonant_diacritic','grapheme_root', 'vowel_diacritic']
HEIGHT = 137
WIDTH = 236


reHEIGHT =128
reWIDTH =128
N_CHANNELS=3


# In[3]:


def resize(df):
    resized = {}        
    for i in range(df.shape[0]):
        
        image = df.loc[df.index[i]].values.reshape(137,236)   
        #image=crop_char_image(image,128)
        image=cv2.resize(image,(reWIDTH,reHEIGHT),interpolation=cv2.INTER_AREA)
        m, s = cv2.meanStdDev(image)
        m, s=m[0][0],s[0][0]
        image=(image-m)/(s+1e-9)           
        resized[df.index[i]] = image.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized
def swish(x):
    return (K.sigmoid(x) * x)
class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


# In[4]:



#model=tfkeras.models.load_model('/kaggle/input/v19-avg3efficientnetb3-13-mixup25-cutout50-128-128/avg3EfficientNetB3_13.h5',custom_objects={'swish':swish,'FixedDropout':FixedDropout})
model=tfkeras.models.load_model('/kaggle/input/v19-avg3seresnet34-7-mixup25-cutout50-128-128/avg3seresnet34_7.h5')
#model.summary()


# In[5]:


#single model

target=[] # model predictions placeholder
row_id=[] # row_id place holder
cnt=0
for i in range(4):
    
    #start=time.clock()
    #print(start)
    X_test = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i))#用pandas读取parquet格式的数据

    X_test.set_index('image_id', inplace=True)#DataFrame可以通过set_index方法，可以设置单索引和复合索引。inplace为True时，索引将会还原为列

    LenX_test=X_test.shape[0]#读取数据大小
    batchsize=min(LenX_test,1024)#与1024比，哪个小，把批量大小换做哪个，适用于比赛
    #tqdm.pandas(desc="my bar!")#为了显示进度条写的
    for each in range(0,LenX_test,batchsize):
        batchsize=min(LenX_test-each,1024)#批量大小重新被赋值，如果一个batchsize没有1024大，那么用LenX_test-each差值刷新批量大小。适用于比赛。
        X_testbc=X_test[each:each+batchsize]#取一段test数据
        X_testbc=resize(X_testbc).values.reshape(-1, reHEIGHT,reWIDTH) #由于内存限制，所以取一段test数据来进行预测               
        temp1,temp2,temp3=0,0,0 #临时预测变量
        predict=model.predict(np.array([np.repeat((X_testbc[y]),3) for y in range(batchsize)]).reshape(-1,reHEIGHT,reWIDTH,N_CHANNELS),batch_size=batchsize,use_multiprocessing=False)   
        temp1=temp1+predict[0]#
        temp2=temp2+predict[1]
        temp3=temp3+predict[2]
        del predict#垃圾回收机制是：先删除。
        gc.collect()#再回收。
        
        for x in range(batchsize): 
            row_id.append('Test_'+str(cnt)+'_consonant_diacritic')
            target.append(np.argmax(temp1[x], axis=0))#取最大值的索引。
            row_id.append('Test_'+str(cnt)+'_grapheme_root')
            target.append(np.argmax(temp2[x], axis=0))
            row_id.append('Test_'+str(cnt)+'_vowel_diacritic')
            target.append(np.argmax(temp3[x], axis=0))
            cnt+=1
       
        del X_testbc
        del temp1
        del temp2
        del temp3
        gc.collect()  
        

    #print("runtime is:",time.clock()-start)

    #del df_test_img
    del X_test
    gc.collect()
df_sample = pd.DataFrame(
    {
        'row_id': row_id,
        'target':target
    },
    columns = ['row_id','target'] 
)
df_sample.to_csv('submission.csv',index=False)

