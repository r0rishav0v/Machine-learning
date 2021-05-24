# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and setting
import pandas as pd
import numpy as np



def Kushkal_Wallis(df):
    df.columns = ['Categorical','Numerical']
    unique_categories = df['Categorical'].unique()
    my_dict = dict()
    k=0
    for i in unique_categories:
        Data = df[df['Categorical']==i]
        #my_dict = dict([(i, sum(Data['Numerical'], Data['Numerical'].count()))])
        my_dict[i] =  [sum(Data['Numerical']),Data['Numerical'].count()]
    C = sum(df['Numerical']) / df['Numerical'].count()
    for i in unique_categories:
        z =  my_dict[i][0]**2
        k= k+z
    SS_Total = k-C
    k = 0
    for i in unique_categories:
        z =  (my_dict[i][0]**2)/my_dict[i][1])
        k= k+z
    SS_Between = k - C
    SS_within = SS_Total - SS_Between
    DF_Total = df['Numerical'].count() - 1
    DF_BTW = len(unique_categories) - 1
    DF_within = DF_Total - DF_BTW
    MSS_BTW = SS_Between/DF_BTW
    MSS_WITHIN = SS_within/DF_within
    F_ratio = MSS_BTW/MSS_WITHIN
    return my_dict, F_ratio, DF_BTW,DF_within

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    z,c,df_btw,df_within = Kushkal_Wallis(Data[['SaleCondition','SalePrice']])
    print("F ratio : ", c)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
