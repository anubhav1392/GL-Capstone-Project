#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as pyd
from tqdm import tqdm


# In[2]:


images_path='../input/rsna-pneumonia-detection-challenge/stage_2_train_images'
train_labels_df=pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv')
label_meta_data=pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')


# In[3]:


#Lets take a look at size of our dataset and how many unique Dicom images are there
print('Size of Dataset: ',train_labels_df.shape)
print('Number of Unique X-Rays: ',train_labels_df['patientId'].nunique())


# In[4]:


#lets take a look at our Target Distribution
label_count=train_labels_df['Target'].value_counts()
explode = (0.1,0.0)  

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(label_count.values, explode=explode, labels=['Normal','Pneumonia'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.title('Target Distribution')
plt.show()


# In[5]:


#Class distribution for target
label_count=label_meta_data['class'].value_counts()
explode = (0.01,0.01,0.01)  

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(label_count.values, explode=explode, labels=label_count.index, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal') 
plt.title('Class Distribution')
plt.show()


# In[40]:


#Let's plot some images along with their target
r=c=3
fig=plt.figure(figsize=(15,14))
for i in range(1,r*c+1):
    id_=np.random.choice(train_labels_df['patientId'].values)
    label_0=np.unique(train_labels_df['Target'][train_labels_df['patientId']==id_])
    label_1=np.unique(label_meta_data['class'][label_meta_data['patientId']==id_])
    
    #read xray
    img=pyd.read_file(os.path.join(images_path,id_+'.dcm')).pixel_array
    fig.add_subplot(r,c,i)
    plt.imshow(img,cmap='gray')
    if label_0==1:
        plt.title('Pneumonia Infected'+' | '+label_1)
    else:
        plt.title('Normal Xray'+' | '+label_1)
    plt.xticks([])
    plt.yticks([])


# In[42]:


print('Dtype of Xrays: ',img.dtype)


# In[7]:


# Plot image with bounding box
id_='00436515-870c-4b36-a041-de91049b9ab4' #select random id
class_=label_meta_data['class'][label_meta_data['patientId']==id_]

plt.figure(figsize=(15,10))
current_axis = plt.gca()
img=pyd.read_file(os.path.join(images_path,id_+'.dcm')).pixel_array
plt.imshow(img,cmap='bone')


current_axis = plt.gca()
boxes=train_labels_df[['x','y','width','height']][train_labels_df['patientId']==id_].values

for box in boxes:
    x=box[0]
    y=box[1]
    w=box[2]
    h=box[3]
    current_axis.add_patch(plt.Rectangle((x, y), w, h, 
                                         color='green', fill=False, linewidth=2))  
    #current_axis.text(x,y, class_, size='x-large', 
    #                  color='white', bbox={'facecolor':'green', 'alpha':1.0})


# In[11]:


#Get Age,target,sex meta data for each unique patient
age_df=pd.DataFrame(columns=['age','sex','target'])

for ix,id_ in tqdm(enumerate(train_labels_df['patientId'].unique())):
    age=pyd.read_file(os.path.join(images_path,
                                   id_+'.dcm')).PatientAge
    s=pyd.read_file(os.path.join(images_path,
                                   id_+'.dcm')).PatientSex
    
    t=train_labels_df['Target'][train_labels_df['patientId']==id_].unique()

    age_df.loc[ix,'age']=age
    age_df.loc[ix,'sex']=s
    age_df.loc[ix,'target']=t[0]


# In[38]:


#let's see how many males and females have pneumonia and normal
tmp=pd.DataFrame(age_df.groupby(['sex','target']).count())
tmp.rename(columns={'age':'count'},inplace=True)
tmp.plot.bar()
plt.ylabel('Count')
plt.grid()
plt.show()

