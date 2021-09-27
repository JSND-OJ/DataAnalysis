#!/usr/bin/env python
# coding: utf-8

# # 생체 광학 데이터 분석 AI 경진대회

# https://dacon.io/competitions/official/235608/overview/description

# In[1]:


# 판다스와 넘파이
import pandas as pd
import numpy as np

# 이미지, 시각화, 분석
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 머신러닝(인공지능), ligtgbm, 선형회귀
import lightgbm as lgb
from lightgbm import LGBMRegressor

# 검증평가
from sklearn.multioutput import MultiOutputRegressor # 두개 이상의 label을 가질 경우 
from sklearn.model_selection import cross_val_score # 교차 검증
from sklearn.metrics import mean_absolute_error # loss 함수 MAE

# 반복문 시간을 알려주는것, for 반복문 쓴거 보다 10배 빠름.
from tqdm import tqdm
# 불필요한 오류를 안보이게 해줌
import warnings ; warnings.filterwarnings('ignore') 


# ##  데이터 불러오기

# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()


# In[3]:


# id : index number랑 동일 필요 없음.
# rho : 측정거리(mm) 25, 10, 15, 20
train['rho'].unique()


# In[4]:


train.info()


# In[5]:


# 시각화를 통해 빈값을 한눈에 알아 볼 수 있다.
train.isnull().sum().plot()


# In[6]:


train.columns


# In[7]:


column = ['650_dst', '660_dst', '670_dst', '680_dst', '690_dst', '700_dst',
          '710_dst', '720_dst', '730_dst', '740_dst', '750_dst', '760_dst',
          '770_dst', '780_dst', '790_dst', '800_dst', '810_dst', '820_dst',
          '830_dst', '840_dst', '850_dst', '860_dst', '870_dst', '880_dst',
          '890_dst', '900_dst', '910_dst', '920_dst', '930_dst', '940_dst',
          '950_dst', '960_dst', '970_dst', '980_dst', '990_dst']
train[column].head()


# In[8]:


train_dst = train.filter(regex = '_dst$', axis='columns')
train_dst.columns


# In[9]:


train_dst.isnull().sum().plot()


# In[10]:


train_dst.head().T


# In[11]:


train_dst.head().T.plot()


# In[12]:


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,6)
train.filter(regex = '_src$', axis='columns').head().T.plot(ax=ax1)
train.filter(regex = '_dst$', axis='columns').head().T.plot(ax=ax2)


# ## 보간 하기

# In[13]:


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,6)
train.filter(regex = '_dst$', axis='columns').head().T.plot(ax=ax1)
train.filter(regex = '_dst$', axis='columns').head().T.interpolate(method='linear',axis=0).plot(ax=ax2)


# In[14]:


train_dst.columns


# In[15]:


list(650 + np.arange(35)*10)


# In[16]:


train_dst.columns = list(650 + np.arange(35)*10)
train_dst.columns 


# In[17]:


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,6)
train_dst.head().T.interpolate(method='linear',axis=0).plot(ax=ax1)
train_dst.head().T.interpolate(method='cubic',axis=0).plot(ax=ax2)


# In[18]:


# linear, nearest, zero, slinear, quadratic, cubic, spline, polynomial
figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_dst.head().T.interpolate(method='linear',axis=0).plot(ax=ax1)
train_dst.head().T.interpolate(method='nearest',axis=0).plot(ax=ax2)
train_dst.head().T.interpolate(method='zero',axis=0).plot(ax=ax3)
train_dst.head().T.interpolate(method='slinear',axis=0).plot(ax=ax4)
train_dst.head().T.interpolate(method='quadratic',axis=0).plot(ax=ax5)
train_dst.head().T.interpolate(method='cubic',axis=0).plot(ax=ax6)
train_dst.head().T.interpolate(method='spline',order=3,axis=0).plot(ax=ax7)
train_dst.head().T.interpolate(method='polynomial',order=3,axis=0).plot(ax=ax8)


# In[19]:


from scipy.interpolate import interp1d

x = np.linspace(0, 10, num=11, endpoint = True)
y = np.cos(-x**2/9)
f = interp1d(x,y) # linear
f2 = interp1d(x,y, kind = 'nearest')
f3 = interp1d(x,y, kind = 'zero')
f4 = interp1d(x,y, kind = 'quadratic')
f5 = interp1d(x,y, kind = 'cubic')

xnew = np.linspace(0, 10, num = 41, endpoint = True)

plt.figure(figsize=(18,8))
plt.plot(x,y,'o', xnew, f(xnew),'-', xnew, f2(xnew),'--', xnew, f3(xnew),'.-', 
         xnew, f4(xnew),'-.', xnew, f5(xnew),':')
plt.legend(['data','linear','nearest','zero','quadratic','cubic'])
plt.show()


# In[20]:


train_dst.columns = list(650 + np.arange(35)*10) # 문자열 컬럼을 숫자 컬럼으로 변경
train_dst = train_dst.interpolate(method='polynomial',order=3,axis=1) # polynomial 보간으로 빈값 채움
train_dst_i = train_dst.interpolate(method='linear',axis=1) # linear 보간으로 빈값 채움
train_dst_i.columns=train.filter(regex='_dst$', axis='columns').columns #숫자 컬럼을 문자열 컬럼으로 되돌림
train_dst_i.columns


# In[21]:


test_dst = test.filter(regex='_dst', axis='columns') # _dst 컬럼 추출
test_dst.columns = list(650 + np.arange(35)*10) # 문자열 컬럼 숫자 컬럼 변환
test_dst = test_dst.interpolate(method='polynomial',order=3,axis=1) # polynomial 보간으로 빈값 채움
test_dst_i = test_dst.interpolate(method='linear',axis=1) # linear 보간
test_dst_i.columns = test.filter(regex='_dst$', axis='columns').columns #숫자 컬럼을 문자열 컬럼으로 변경
test_dst_i.columns


# In[22]:


figure, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(18,8)
train_dst_i.head().T.plot(ax=ax1)
test_dst_i.head().T.plot(ax=ax2)
train_dst_i.isnull().sum().plot(ax=ax3)
test_dst_i.isnull().sum().plot(ax=ax4)


# In[23]:


# 오른쪽 값으로 보간
# 문자열에 수식 입력 i =0이면'980_dst', i=1이면 '970_dst'...
for i in range(34):
    train_dst_i.loc[train_dst_i[f'{980-(i*10)}_dst'].isnull(), 
                    f'{980-(i*10)}_dst'] = train_dst_i.loc[train_dst_i[f'{980-(i*10)}_dst'].isnull(),
                                                           f'{990-(i*10)}_dst']
    test_dst_i.loc[test_dst_i[f'{980-(i*10)}_dst'].isnull(), 
                   f'{980-(i*10)}_dst'] = test_dst_i.loc[test_dst_i[f'{980-(i*10)}_dst'].isnull(),
                                                         f'{990-(i*10)}_dst']


# In[24]:


figure, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(18,8)
train_dst_i.head().T.plot(ax=ax1)
test_dst_i.head().T.plot(ax=ax2)
train_dst_i.isnull().sum().plot(ax=ax3)
test_dst_i.isnull().sum().plot(ax=ax4)


# In[25]:


# train/test data를 update 함  
train.update(train_dst_i)
test.update(test_dst_i)
train


# In[26]:


test


# In[27]:


figure, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(18,8)
train.head().T.plot(ax=ax1)
test.head().T.plot(ax=ax2)
train.isnull().sum().plot(ax=ax3)
test.isnull().sum().plot(ax=ax4)


# ## 흡광도 A컬럼 만들기

# In[28]:


# beer-lamber 법칙 --> 흡광도 A 컬럼 만들기
# np.log(src/dst) ratio2
# np.log(src/dst)/(rho*0.1) ratio
for i in range(35):
    train[f'{650+(i*10)}_ratio']=np.log(train[f'{650+(i*10)}_src']/train[f'{650+(i*10)}_dst'])/(train['rho']*0.1)
    test[f'{650+(i*10)}_ratio']=np.log(test[f'{650+(i*10)}_src']/test[f'{650+(i*10)}_dst'])/(test['rho']*0.1)


# In[29]:


for i in range(35):
    train[f'{650+(i*10)}_ratio2']=np.log(train[f'{650+(i*10)}_src']/train[f'{650+(i*10)}_dst'])
    test[f'{650+(i*10)}_ratio2']=np.log(test[f'{650+(i*10)}_src']/test[f'{650+(i*10)}_dst'])


# In[30]:


# ratio의 그래프, 빈값, 음의 무한대, 양의 무한대
figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train.filter(regex='_ratio$', axis='columns').head(10).T.plot(ax=ax1)
test.filter(regex='_ratio$', axis='columns').head(10).T.plot(ax=ax2)
train.filter(regex='_ratio$', axis='columns').isnull().sum().plot(ax=ax3)
test.filter(regex='_ratio$', axis='columns').isnull().sum().plot(ax=ax4)
train.filter(regex='_ratio$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax5)
test.filter(regex='_ratio$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax6)
train.filter(regex='_ratio$', axis='columns').isin([float('inf')]).sum().plot(ax=ax7)
test.filter(regex='_ratio$', axis='columns').isin([float('inf')]).sum().plot(ax=ax8)


# In[31]:


# ratio의 그래프, 빈값, 음의 무한대, 양의 무한대
figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train.filter(regex='_ratio2$', axis='columns').head().T.plot(ax=ax1)
test.filter(regex='_ratio2$', axis='columns').head().T.plot(ax=ax2)
train.filter(regex='_ratio2$', axis='columns').isnull().sum().plot(ax=ax3)
test.filter(regex='_ratio2$', axis='columns').isnull().sum().plot(ax=ax4)
train.filter(regex='_ratio2$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax5)
test.filter(regex='_ratio2$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax6)
train.filter(regex='_ratio2$', axis='columns').isin([float('inf')]).sum().plot(ax=ax7)
test.filter(regex='_ratio2$', axis='columns').isin([float('inf')]).sum().plot(ax=ax8)


# In[32]:


# 음의 무한대와 양의 무한대를 np.nan로 변경
#train_r=train.filter(regex='_ratio$',axis='columns').replace(float('-inf'),np.nan).replace(float('inf'),np.nan)
#test_r=test.filter(regex='_ratio$',axis='columns').replace(float('-inf'),np.nan).replace(float('inf'),np.nan)

#train_r2=train.filter(regex='_ratio2$',axis='columns').replace(float('-inf'),np.nan).replace(float('inf'),np.nan)
#test_r2=test.filter(regex='_ratio2$',axis='columns').replace(float('-inf'),np.nan).replace(float('inf'),np.nan)

train_r = train.replace(float('-inf'),np.nan).replace(float('inf'),np.nan)
test_r=test.replace(float('-inf'),np.nan).replace(float('inf'),np.nan)


# In[33]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r.filter(regex='_ratio$', axis='columns').head(10).T.plot(ax=ax1)
test_r.filter(regex='_ratio$', axis='columns').head(10).T.plot(ax=ax2)
train_r.filter(regex='_ratio$', axis='columns').isnull().sum().plot(ax=ax3)
test_r.filter(regex='_ratio$', axis='columns').isnull().sum().plot(ax=ax4)
train_r.filter(regex='_ratio$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax5)
test_r.filter(regex='_ratio$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax6)
train_r.filter(regex='_ratio$', axis='columns').isin([float('inf')]).sum().plot(ax=ax7)
test_r.filter(regex='_ratio$', axis='columns').isin([float('inf')]).sum().plot(ax=ax8)


# In[34]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r.filter(regex='_ratio2$', axis='columns').head(10).T.plot(ax=ax1)
test_r.filter(regex='_ratio2$', axis='columns').head(10).T.plot(ax=ax2)
train_r.filter(regex='_ratio2$', axis='columns').isnull().sum().plot(ax=ax3)
test_r.filter(regex='_ratio2$', axis='columns').isnull().sum().plot(ax=ax4)
train_r.filter(regex='_ratio2$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax5)
test_r.filter(regex='_ratio2$', axis='columns').isin([float('-inf')]).sum().plot(ax=ax6)
train_r.filter(regex='_ratio2$', axis='columns').isin([float('inf')]).sum().plot(ax=ax7)
test_r.filter(regex='_ratio2$', axis='columns').isin([float('inf')]).sum().plot(ax=ax8)


# ## A컬럼 보간하기

# In[35]:


# ratio와 ratio2 간 인접한 컬럼끼리 보간되는 것을 막기 위해 분리함
train_r1 = train_r.filter(regex='_ratio$', axis='columns')
train_r2 = train_r.filter(regex='_ratio2$', axis='columns')

test_r1 = test_r.filter(regex='_ratio$', axis='columns')
test_r2 = test_r.filter(regex='_ratio2$', axis='columns')


# In[36]:


# 문자열 컬럼을 숫자 컬럼으로 변경
train_r1.columns = list(650 + 10*np.arange(35))
train_r2.columns = list(650 + 10*np.arange(35))
test_r1.columns = list(650 + 10*np.arange(35))
test_r2.columns = list(650 + 10*np.arange(35))


# In[37]:


# polynomial 보간
train_r1 = train_r1.interpolate(method='polynomial',order=3, axis=1)
train_r2 = train_r2.interpolate(method='polynomial',order=3, axis=1)

test_r1 = test_r1.interpolate(method='polynomial',order=3, axis=1)
test_r2 = test_r2.interpolate(method='polynomial',order=3, axis=1)


# In[38]:


# linear 보간
train_r1 = train_r1.interpolate(method='linear', axis=1)
train_r2 = train_r2.interpolate(method='linear', axis=1)

test_r1 = test_r1.interpolate(method='linear', axis=1)
test_r2 = test_r2.interpolate(method='linear', axis=1)


# In[39]:


# 숫자 컬럼을 문자 컬럼으로 변경
train_r1.columns = train.filter(regex='_ratio$', axis='columns').columns
train_r2.columns = train.filter(regex='_ratio2$', axis='columns').columns

test_r1.columns = test.filter(regex='_ratio$', axis='columns').columns
test_r2.columns = test.filter(regex='_ratio2$', axis='columns').columns

test_r2.columns


# In[40]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r1.head(10).T.plot(ax=ax1)
test_r1.head(10).T.plot(ax=ax2)
train_r1.isnull().sum().plot(ax=ax3)
test_r1.isnull().sum().plot(ax=ax4)
train_r1.isin([float('-inf')]).sum().plot(ax=ax5)
test_r1.isin([float('-inf')]).sum().plot(ax=ax6)
train_r1.isin([float('inf')]).sum().plot(ax=ax7)
test_r1.isin([float('inf')]).sum().plot(ax=ax8)


# In[41]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r2.head(10).T.plot(ax=ax1)
test_r2.head(10).T.plot(ax=ax2)
train_r2.isnull().sum().plot(ax=ax3)
test_r2.isnull().sum().plot(ax=ax4)
train_r2.isin([float('-inf')]).sum().plot(ax=ax5)
test_r2.isin([float('-inf')]).sum().plot(ax=ax6)
train_r2.isin([float('inf')]).sum().plot(ax=ax7)
test_r2.isin([float('inf')]).sum().plot(ax=ax8)


# In[42]:


# 오른쪽 보간
# ratio1
for i in range(34):
    train_r1.loc[train_r1[f'{980-(i*10)}_ratio'].isnull(),
                 f'{980-(i*10)}_ratio']=train_r1.loc[train_r1[f'{980-(i*10)}_ratio'].isnull(),
                                                     f'{990-(i*10)}_ratio']
    test_r1.loc[test_r1[f'{980-(i*10)}_ratio'].isnull(),
                 f'{980-(i*10)}_ratio']=test_r1.loc[test_r1[f'{980-(i*10)}_ratio'].isnull(),
                                                     f'{990-(i*10)}_ratio']

# ratio2    
for i in range(34):
    train_r2.loc[train_r2[f'{980-(i*10)}_ratio2'].isnull(),
                 f'{980-(i*10)}_ratio2']=train_r2.loc[train_r2[f'{980-(i*10)}_ratio2'].isnull(),
                                                     f'{990-(i*10)}_ratio2']
    test_r2.loc[test_r2[f'{980-(i*10)}_ratio2'].isnull(),
                 f'{980-(i*10)}_ratio2']=test_r2.loc[test_r2[f'{980-(i*10)}_ratio2'].isnull(),
                                                     f'{990-(i*10)}_ratio2']


# In[43]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r1.head(10).T.plot(ax=ax1)
test_r1.head(10).T.plot(ax=ax2)

train_r1.isnull().sum().plot(ax=ax3)
test_r1.isnull().sum().plot(ax=ax4)

train_r1.isin([float('-inf')]).sum().plot(ax=ax5)
test_r1.isin([float('-inf')]).sum().plot(ax=ax6)

train_r1.isin([float('inf')]).sum().plot(ax=ax7)
test_r1.isin([float('inf')]).sum().plot(ax=ax8)


# In[44]:


figure, ((ax1, ax2),(ax3, ax4),(ax5, ax6),(ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
figure.set_size_inches(18,16)
train_r2.head(10).T.plot(ax=ax1)
test_r2.head(10).T.plot(ax=ax2)

train_r2.isnull().sum().plot(ax=ax3)
test_r2.isnull().sum().plot(ax=ax4)

train_r2.isin([float('-inf')]).sum().plot(ax=ax5)
test_r2.isin([float('-inf')]).sum().plot(ax=ax6)

train_r2.isin([float('inf')]).sum().plot(ax=ax7)
test_r2.isin([float('inf')]).sum().plot(ax=ax8)


# In[45]:


train.update(train_r1)
train.update(train_r2)
test.update(test_r1)
test.update(test_r2)

figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,4)

train.isnull().sum().plot(ax=ax1)
test.isnull().sum().plot(ax=ax2)


# # 분석

# In[46]:


train.columns


# In[47]:


# src ,dst 컬럼이 불필요해 보임
# dst, ratio, ratio2 컬럼 모두 hhb예측에 도움이 되지만 ca, na 예측에는 효율이 떨어져 보임
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(18,6)

sns.heatmap(train.corr().loc['rho':'990_dst','hhb':'na'],ax=ax1)
sns.heatmap(train.corr().loc['650_ratio':'990_ratio2','hhb':'na'],ax=ax2)


# In[48]:


# ca, na 농도 예측의 정도를 높일 수 있는 새로운 컬럼을 찾는 것
# G 값 제거를 위해 retio, ratio2 컬럼 파장 별로 빼기를 한 새로운 컬럼

# ratio 빼기
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff'] = train[f'{650+10*i}_ratio'] - train['990_ratio']
    test[f'{650+(10*i)}_ratio_diff'] = test[f'{650+10*i}_ratio'] - test['990_ratio']
    
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff2'] = train[f'{650+10*i}_ratio'] - (train['850_ratio']+train['860_ratio'])/2
    test[f'{650+(10*i)}_ratio_diff2'] = test[f'{650+10*i}_ratio'] - (test['850_ratio']+test['860_ratio'])/2


# In[49]:


# ratio2 빼기

for i in range(35):
    train[f'{650+(10*i)}_ratio2_diff'] = train[f'{650+10*i}_ratio2'] - train['810_ratio2']
    test[f'{650+(10*i)}_ratio2_diff'] = test[f'{650+10*i}_ratio2'] - test['810_ratio2']


# In[50]:


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(20,10)

sns.heatmap(train.corr().loc['650_ratio_diff':'990_ratio_diff','hhb':'na'],ax=ax1)
sns.heatmap(train.corr().loc['650_ratio_diff2':'990_ratio_diff2','hhb':'na'],ax=ax2)


# In[51]:


# ratio2_diff 컬럼은 ca 농도 예측에 도움이 될 것으로 보임
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(20,10)

sns.heatmap(train.corr().loc['650_ratio2':'990_ratio2','hhb':'na'],ax=ax1)
sns.heatmap(train.corr().loc['650_ratio2_diff':'990_ratio2_diff','hhb':'na'],ax=ax2)


# In[54]:


# 뻬기 실습을 해보세요. 790~830_ratio 범위를 빼보세요
# ratio 빼기
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff3'] = train[f'{650+10*i}_ratio'] - train['810_ratio']
    test[f'{650+(10*i)}_ratio_diff3'] = test[f'{650+10*i}_ratio'] - test['810_ratio']
    
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff4'] = train[f'{650+10*i}_ratio'] - (train['800_ratio']+train['810_ratio'])/2
    test[f'{650+(10*i)}_ratio_diff4'] = test[f'{650+10*i}_ratio'] - (test['800_ratio']+test['810_ratio'])/2


# In[55]:


for i in range(35):
    train[f'{650+(10*i)}_ratio_diff5'] = train[f'{650+10*i}_ratio'] - train['820_ratio']
    test[f'{650+(10*i)}_ratio_diff5'] = test[f'{650+10*i}_ratio'] - test['820_ratio']
    
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff6'] = train[f'{650+10*i}_ratio'] - (train['800_ratio']+train['810_ratio'])/2
    test[f'{650+(10*i)}_ratio_diff6'] = test[f'{650+10*i}_ratio'] - (test['800_ratio']+test['810_ratio'])/2


# In[56]:


for i in range(35):
    train[f'{650+(10*i)}_ratio_diff7'] = train[f'{650+10*i}_ratio'] - train['810_ratio']
    test[f'{650+(10*i)}_ratio_diff7'] = test[f'{650+10*i}_ratio'] - test['810_ratio']
    
for i in range(35):
    train[f'{650+(10*i)}_ratio_diff8'] = train[f'{650+10*i}_ratio'] - (train['800_ratio']+train['810_ratio'])/2
    test[f'{650+(10*i)}_ratio_diff8'] = test[f'{650+10*i}_ratio'] - (test['800_ratio']+test['810_ratio'])/2


# In[57]:


figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(20,12)

sns.heatmap(train.corr().loc['650_ratio_diff3':'990_ratio_diff3','hhb':'na'],ax=ax1)
sns.heatmap(train.corr().loc['650_ratio_diff4':'990_ratio_diff4','hhb':'na'],ax=ax2)


# In[64]:


# ratio를 평균으로 나눠서 그 비율로 새로운 컬럼 생성
# 노이즈나 이상치의 형향을 줄여주는 효과
# 많은 회수의 반복문을 쓸 경우 tqdm을 사용하면 계산 시간을 1/10까지 줄일 수 있다
train_ratio = train.filter(regex='_ratio$', axis='columns')
test_ratio = test.filter(regex='_ratio$', axis='columns')

for i in tqdm(train_ratio.index):
    train_ratio.loc[i] = train_ratio.loc[i] / train_ratio.loc[i].replace(0,np.nan).mean()
    
for i in tqdm(train_ratio.index):
    test_ratio.loc[i] = test_ratio.loc[i] / test_ratio.loc[i].replace(0,np.nan).mean()
    
for i in range(35):
    train[f'{650+(10*i)}_ratio_m'] = train_ratio[f'{650+(10*i)}_ratio']
    test[f'{650+(10*i)}_ratio_m'] = test_ratio[f'{650+(10*i)}_ratio'] 


# In[72]:


for i in range(35):
    train[f'{650+10*i}_ratio_m_d'] = train[f'{650+10*i}_ratio_m'] - (train['810_ratio_m']+train['820_ratio_m'])/2
    test[f'{650+10*i}_ratio_m_d'] = test[f'{650+10*i}_ratio_m'] - (test['810_ratio_m']+test['820_ratio_m'])/2
                                         
figure, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
figure.set_size_inches(20,12)

sns.heatmap(train.corr().loc['650_ratio_m':'990_ratio_m','hhb':'na'],ax=ax1)
sns.heatmap(train.corr().loc['650_ratio_m_d':'990_ratio_m_d','hhb':'na'],ax=ax2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




