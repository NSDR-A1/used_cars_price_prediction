

"""### Data Pre-processing"""

# Commented out IPython magic to ensure Python compatibility.
# Importing Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings('ignore')

from matplotlib import style
style.use('ggplot')


df=pd.read_excel('data/vehicles_Manheim.xlsx', sheet_name='Sheet1')
df=pd.DataFrame(df)


# Getting Rid of Irrelevant Rows
df2=df.copy()
#print('Shape of dataframe before: ', df2.shape)
df2 = df2[df2.isnull().sum(axis=1) > 4]
#print('Shape of dataframe after: ', df2.shape)


# Drop Columns and Check Missing Values
df2=df2.drop(columns=['Subseries', 'Drs','Cyl','Fuel','EW','Radio','Int'])
#df2.head()

#return series of columns with respective of number of null values
#df2.isnull().sum()

#heatmap to identify nulll values using graph
#sns.heatmap(df2.isnull(),yticklabels=False,cbar=True,cmap='Accent')

# Getting Desired Data
df3 = df2[pd.notnull(df2['Price'])]
df3 = df3[pd.notnull(df3['Top'])]
df3 = df3[pd.notnull(df3['4x4'])]
df3 = df3[pd.notnull(df3['Trans'])]
df3 = df3[pd.notnull(df3['Color'])]
df3 = df3[pd.notnull(df3['Model'])]

#df3.isnull().sum()

# Defining the variable categories
#define numeric variable and categorical variable to work separatly on them
num_col=['Year', 'Odometer']
cat_cols=['Make','Model','Color','Trans','4x4','Top']

# Making strings and Integers
for cat in cat_cols:
  df3[cat] = df3[cat].fillna(-1)
  df3[cat] = df3[cat].astype(str)
  df3[cat] = df3[cat].replace('-1', np.nan)

#df3.isnull().sum()

# Making strings and Integers
for num in num_col:
  df3[num] = df3[num].astype(int)


# Saving the processed data
df3.to_csv(r'data/vehicles_Manheim_cleaned.csv',index=False)
df3=pd.read_csv('data/vehicles_Manheim_cleaned.csv')

"""### Outliers"""

#outliers_condi=Latex(r" $\textbf{W𝑒 𝑐𝑎𝑛 𝑠𝑎𝑦 $𝑥_1$ or $x_2$ 𝑖𝑠 𝑜𝑢𝑡𝑙𝑖𝑒𝑟𝑠 if }\\ x_1 < Q1 - 1.5*IQR \\ or\\ x_2 > Q3+1.5*IQR $")
#outliers_info=Latex(r"$L_{p} = \frac{p}{100}(n+1) = i_p.f_p \\ where \,\, i_p \,\, is \,\, integer \,\, part \,\, of \,\, L_p \,\, and \,\, f_p \,\, is \,\, fractional \,\, part \,\, of \,\, L_p \\ Q1 = Y_{25} = x_{i_p} + f_p*(x_{i_{p+1}}-x_{i_p}) \\ Q3 = Y_{75} = x_{i_p} + f_p*(x_{i_{p+1}}-x_{i_p}) \\ IQR = Q3-Q1 \\ x_1 = Q1 - 1.5*IQR \,\,and\,\, x_2 = Q3+1.5*IQR $")

# It will return the range of the variables and the values outside this range will be outliers
def outliers(arr,col):
    x=sorted(arr[col].values.ravel())
    L_25=25/100*(len(x)+1) #L_p where p=25%
    i_p=int(str(L_25).split(".")[0])
    f_p=int(str(L_25).split(".")[1])
    q1=x[i_p]+f_p*(x[i_p+1]-x[i_p])
    
    L_75=75/100*(len(x)+1) #L_p where p=75%
    i_p=int(str(L_75).split(".")[0])
    f_p=int(str(L_75).split(".")[1])
    q3=x[i_p]+f_p*(x[i_p+1]-x[i_p])
    
    #q1,q3=(arr[col].quantile([0.25,0.75]))
    
    IQR=q3-q1
    x1=q1-1.5*IQR
    x2=q3+1.5*IQR
    return (x1,x2)

# Price

def min_max_price(df):
    r=[]
    q1,q3=(df['logprice'].quantile([0.25,0.75]))
    r.append(q1-1.5*(q3-q1))
    r.append(q3+1.5*(q3-q1))
    return (r)


df3['logprice'] = np.log(df3['Price'])
x=df3['logprice']
price_range=list(range(0,int(max(df3['logprice']))+1))
red_square = dict(markerfacecolor='g', marker='s')
plt.boxplot(x, vert=False)
plt.xticks(price_range)
plt.text(min_max_price(df3)[0]-0.3,1.05,str(round(min_max_price(df3)[0],2)))
plt.text(min_max_price(df3)[1]-0.5,1.05,str(round(min_max_price(df3)[1],2)))
plt.title("Figure 1: Box Plot of Price")
plt.savefig('plots/graph-boxplot-price.jpg')
#plt.show()

# Odometer

fig, ax1 = plt.subplots()
ax1.set_title('Figure 2: Box Plot of Odometer')
ax1.boxplot(df3['Odometer'], vert=False, flierprops=red_square)
plt.savefig('plots/graph-boxplot-odometer.jpg')
#plt.show()

# Year

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))

#ploting boxplot
o1,o2=outliers(df3,'Year')
ax1.boxplot(sorted(df3['Year']), vert=False, flierprops=red_square)
ax1.set_xlabel("Years")
ax1.set_title("Figure 3: Box Plot of Year")
ax1.text(o1-8,1.05,str(round(o1,2)))

#ploting histogram
hist,bins=np.histogram(df3['Year'])
n, bins, patches = ax2.hist(x=df3['Year'], bins=bins)
ax2.set_xlabel("Years")
ax2.set_title("Figure 4: Histogram of Year")
for i in range(len(n)):
    if(n[i]>2000):
        ax2.text(bins[i],n[i]+3000,str(n[i]))

plt.tight_layout()
plt.savefig('plots/graph-barplot-histogram-year.jpg',dpi=1200)
#plt.show()

# Removing outliers

df_new=df3.copy()
out=np.array(['logprice','Odometer','Year'])
for col in out:
    o1,o2=outliers(df_new,col)
    df_new=df_new[(df_new[col]>=o1) & (df_new[col]<=o2)]
    print('IQR of',col,'=',o1,o2)
df_new=df_new[df_new['Price']!=0]
df_new.drop('logprice',axis=1,inplace=True)
#df_new.head()

# Saving the final dataframe
print("Shape before process=",df.shape)
print("Shape After process=",df_new.shape)
diff=df.shape[0]-df_new.shape[0]
print("Total {} rows and {} cols removed".format(diff,df.shape[1]-df_new.shape[1]))
df_new.to_csv("data/vehicles_Manheim_Final.csv",index=False)

