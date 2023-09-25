#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import difflib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:

class movie_recc():
    def main(self,data):
        raw_df = pd.read_csv('D:/Users/PANKAJ/Documents/PANKAJ2.0/PYTHON/PYTHON PROJECT/MOVIERECOMONDED/movie_database.csv')
        #raw_df


        # In[3]:


        genres = raw_df.genres.replace('\|'," ",regex=True)


        # In[4]:


        genres = raw_df.genres.replace('-',"",regex=True)


        # In[5]:


        genres = raw_df.genres.replace('(no genres listed)',"nogenreslisted",regex=True)


        # In[6]:


        cv = CountVectorizer()


        # In[7]:


        genres_array = cv.fit_transform(genres).toarray()


        # In[8]:


        genres_name = cv.get_feature_names_out()


        # In[9]:


        genres_data = pd.DataFrame(data=genres_array,columns=genres_name)
        #genres_data


        # In[11]:


        lb = LabelEncoder()


        # In[12]:


        company_lb = lb.fit_transform(raw_df.company)


        # In[13]:


        temp = raw_df[['movieId','title']]
        #temp


        # In[14]:


        df= pd.concat([temp,genres_data],axis=1)
        #df


        # In[15]:


        df['company'] = company_lb
        df['score'] = raw_df.score



        # In[16]:


        self.df2 = df.drop_duplicates()
        #df2


        # In[18]:


        self.df3 = self.df2.drop(columns=['movieId','title'],axis=1)
        #df3


        # In[19]:


        self.knn = NearestNeighbors()


        # In[20]:


        self.knn.fit(self.df3)


        return self.ext_ind(data)


    def ext_ind(self,name):
        x = difflib.get_close_matches(name,self.df2.title)
        ind = self.df2[self.df2.title==x[0]].index[0]
        return self.using_knn(ind)
            
    def using_knn(self,ind):
        distance,index = self.knn.kneighbors(self.df3.iloc[[ind]],n_neighbors=10)
        movie_list = []
        for i in index[0]:
            if ind!=i:
                y = self.df2[self.df2.index==i]['title'].values
                z = "".join(y)

                movie_list.append(z)
            
        return movie_list
        
        
            


# In[23]:


#x = movie_recc()
#res = "\n".join(x.main('toy story'))

#print(res)
# In[ ]:


if __name__=="__main__":
    print("hi")
    obj = movie_recc()
    x = obj.main("iron man")
    print(x)

