#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Ecem Deniz Babaoğlan
#181201071


# In[ ]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


def relu(x):
    return np.maximum(0,x)


# In[4]:


def deriv_relu(x):
    der=list()
    der.clear()
    for val in x:
        if val>0:
            der.append(1)
        else:
            der.append(0)
    return der


# In[5]:


def initialize(x):
    np.random.seed(12345)
    w1=2*np.random.random((len(x.columns), 1))-1
    bias=2*np.random.random()-1
   # w2=2*np.random.random((neurons, 1))-1
    return w1, bias


# In[6]:


def forward_prop(w1, bias, x):
    x=x.reshape(13,1)
    z=np.dot(w1.T,x)+bias
    out=relu(z)
    return z, out
    


# In[7]:


def back_prop(w1, bias, x, y, z, out, lr):
    x=x.reshape(13,1)
    err=2*(np.subtract(out.sum(axis=0),y))
    dummy=np.array(deriv_relu(z)).reshape(1,1)
    dc_dw=lr*np.dot(x,dummy)*err
    w1=np.subtract(w1,dc_dw)
    dc_db=lr*err
    bias-=dc_db
    return w1, bias


# In[8]:


pd.set_option("display.max_rows", None,
             "display.max_columns", None)


# In[9]:


PATH="/Users/ecemdenizbabaoglan/Desktop/TOBBETU/yap470/housing.csv" #Buraya housing.csv dosyasının bilgisayardaki yolu girilmelidir


# In[10]:


columns=("CRIM","ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV")


# In[11]:


ds=pd.read_csv(PATH, sep='\s+', engine='python', names=columns)


# In[12]:


ds.head()


# In[13]:


target=ds.loc[:,'MEDV'].copy()
data=ds.drop('MEDV', axis=1).copy()


# In[14]:


w1, bias=initialize(data)


# In[15]:


data_normalized = ((data - data.mean()) / data.std(ddof=0)).to_numpy()
target=target.to_numpy()


# In[16]:


cutoff = int(len(data) * 0.8)
x_train, x_test = data_normalized[:cutoff], data_normalized[cutoff:]
y_train, y_test = target[:cutoff], target[cutoff:]


# In[17]:


#for best results without bias: Epochs:2000, lr:0.001
#for best results with bias: Epochs:2000, lr:0.01
#edit: bias ekliyken lr sabit birakilip epoch 500'e kadar düşürüldü, aynı başarım alındı.
#edit2: epoch 100'e düşürüldüğünde training başarımı azaldı (muhtemel overfit engellendi), test başarısı yaklaşık aynı elde edildi


# In[18]:


epochs=50;


# In[19]:


lr=0.01


# In[20]:


train_loss_epoch=list()
test_loss_epoch=list()
out_test_=list()


# In[21]:


for e in range(epochs):
    loss_train=0
    loss_test=0
    out_test_.clear()
    for i in range(len(y_train)):   
        z, out = (forward_prop(w1, bias, x_train[i]))
       # predicted[i]=output[i].sum(axis=0)
        w1, bias=back_prop(w1, bias, x_train[i], y_train[i], z, out, lr)
        loss_train+=np.square(y_train[i]-(out.sum(axis=0)))
        
    train_loss_epoch.append(loss_train/(len(y_train)))   
    
    for i in range(len(y_test)):
        z_test, out_test=forward_prop(w1,bias,x_test[i])
        out_test_.append(out_test)
        loss_test+=np.square(y_test[i]-(out_test.sum(axis=0)))

    test_loss_epoch.append(loss_test/(len(y_test)))


# In[22]:


plt.scatter(range(epochs),train_loss_epoch, color='blue',label='Train Loss')
plt.scatter(range(epochs),test_loss_epoch, color='red', label='Test Loss')
plt.xlabel("epoch")
plt.ylabel("Mean Squared Error")
plt.title("0 Gizli Katman Kayıp Grafiği (lr=0.01)")
plt.legend(loc="upper right")
#plt.savefig("0-hl-lr01.jpg")
plt.show()


# In[23]:


train=sum(train_loss_epoch)/len(train_loss_epoch)
print(train)


# In[24]:


test=sum(test_loss_epoch)/len(test_loss_epoch)
print(test)


# In[25]:


for i in range(len(y_test)):
    print(y_test[i], out_test_[i])

