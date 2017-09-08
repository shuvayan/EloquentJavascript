import subprocess as sp
tmp = sp.call('cls',shell=True)

import numpy as np
import pandas as pd

''' Object Creation '''

## Creating Series: 1D ndarray 
#print(pd.Series( [1, 5, 2, 3, 6, 4] ) )

## Creating DataFrame
#
## 1. Using list of dictionary
#lst = [{"C1": 1, "C2": 2},
#       {"C1": 5, "C2": 10, "C3": 20}]
## Observe NaN       
#print(pd.DataFrame(lst, index = ["R1", "R2"]))
#
## 2. Using list (Commonly used)
#lst = {"C1": ["1", "3"],
#       "C2": ["2","4"]}
#print( pd.DataFrame(lst, index = ["R1", "R2"]) )
#
## Defining Column name within df function
#
#print(pd.DataFrame(np.random.randn(2,2),
#                   index = list('pq'),
#                    columns = list('ab')))


''' Data types '''

#df = pd.DataFrame({'A': [10., 20.],
#                   'B': "text",
#                   'C': [2,60],
#                   'D': 3+9j})
#print(df)
## Observe final dtype as OBJECT to preserve every
## element's data type information.
#print(df.dtypes)
#print(df.info())


''' Viewing Data '''

#print(pd.DataFrame(np.random.randn(20,2)).head())
#print(pd.DataFrame(np.random.randn(20,2)).tail(10))
#
#df = pd.DataFrame({'A': [10., 20.],
#                   'B': "text",
#                   'C': [2,60],
#                   'D': 3+9j}, index = list('PQ'))
#print((df.index))   #index of rows
#print(df.columns)   # index of columns
#print(df.values)    # display values of df



''' DataFrame Attributes '''

#df = pd.DataFrame({'A': [1,4,7],
#                    'B': [2,5,8],
#                    'C': [3,6,9]},
#                    index = list('PQR'))
#print(df)
#
## Transpose
#print(df.T)
#
## shape, size and dimensions
#print(df.shape, df.size, df.ndim)



''' DataFrame Methods '''

#df1 = pd.DataFrame({'A': [1,4,7],
#                    'B': [2,5,8],
#                    'C': [3,6,9]},
#                    index = list('PQR'))
#df_temp = df1
#
#df2 = pd.DataFrame({'A': [-1,4,7],
#                    'B': [2,-5,8],
#                    'C': [3,6,-9]},
#                    index = list('PQR'))
#
#df3 = pd.DataFrame({'A': [1,-4,-7],
#                    'B': [-2,5,-8],
#                    'C': [-3,-6,9]},
#                    index = list('STU'))

## Addition | Subtraction
#print(df1.add(df2))
#print(df1.sub(df3))

## Appending 2 dfs
#print(df_temp.append(df2))

## Applying a function
#print(df3.apply(np.abs, axis = 1))
#print(df3.apply(np.sum, axis = 0))

## Converting frame to its numpy array representation
#print(type(df3.as_matrix()))


## Computing non-nan values
#print(df1.count())  #by default -> col
#print(df1.count(axis = 1))  # row
#print((df1.sub(df3).count()))   # nan in all col


## Commulative operations
#
#print(df3)
#print(df3.cummax(axis = 1))
#print(df3.cummin(axis = 0))
#print(df3.cumsum(axis = 0))
#print(df3.cumprod(axis = 1))


## Check if values are NULL
#df = pd.DataFrame(np.nan,columns = list('AB'),
#                  index = list('CD'))
#print(df)
#print(df.isnull())


## Filling NaNs
#print(df1.sub(df3).fillna(5))


## Drop NaNs
#df = pd.DataFrame([[np.nan, 2, np.nan, 0], 
#                   [3, 4, np.nan, 1],
#                   [np.nan, np.nan, np.nan, 5]],
#                   columns=list('ABCD'))
#print(df)
#print(df.dropna(axis = 1, how = 'all'))
#print(df.dropna(axis = 1, how='any'))


## Handling Duplicates
#
#df = pd.DataFrame([[1, 2, 3], 
#                   [1, 2, 3], 
#                   [4, 5, 6],
#                   [1, 2, 3]], columns =list('ABC'))
## representing which rows are duplicated
#print(df.duplicated())


## Computing PRESENSE of VALUES in a df
#
#df = pd.DataFrame([["hi", 2, 3], 
#                   [1, 20, 3+3j], 
#                   [4, "world", 6]], 
#                    columns =list('ABC'))
#print(df)
#print(df.isin( [20, 6+0j, 'hi'] ))


''' Selection '''

## Selection by Label
#
#df = pd.DataFrame([[15, 12],
#                   [33, 54],
#                   [10, 32]], 
#                   index = list('ABC'),
#                   columns = list('DE'))
#print(df)
#print(df.loc[['A','C'],:])
#print(df.loc[:,'E'])
#print(df.loc['B', 'D'])


## Selection by Position
#
#df = pd.DataFrame([[15, 12],
#                   [33, 54],
#                   [10, 32]])
#
#print(df)
#print(df.iloc[0:2,:]) 
#print(df.iloc[:,0:1])
#print(df.iloc[:,:])  


## Regex
#
#df = pd.DataFrame([[15, 12],
#                   [33, 54],
#                   [10, 32]], 
#                   index = ['one','two','three'],
#                   columns = ['col1', 'col2'])
#
#print(df)
#print(df.filter(regex = 'e$', axis = 0))
#print(df.filter(regex = '^c', axis = 1))

## Boolean indexing
#df = pd.DataFrame([[15, 12],
#                   [33, 54],
#                   [10, 32]])
#
#print(df[df >= 15])




''' Operations '''

#df1 = pd.DataFrame([[15, 12],
#                   [33, 54],
#                   [10, 32]],
#                   columns = list('AB')) 
#df2 = pd.DataFrame([[15, 12, -3],
#                   [33, 54, 21],
#                   [10, 32, 22]],
#                   columns = list('ABC'))
#df3 = pd.DataFrame([[10, 1, 3],
#                   [33, -54, 2],
#                   [10, 0.32, 2]],
#                   columns = list('ABC'))

#print(df1)
#print(df1.eval('B - A'))
#df1.eval('C = B - A')   # < C = > is optional
#print(df1)


## Comparison Operations
#
#print(df1.equals(df2))  # On broader scale
#print(df2.eq(df3))
#print(df2.le(df3))
#print(df2.ne(df3))


## Merging dfs
#
## Concatenating at bottom
#print(pd.concat( [df2,df3], axis = 0 ))
#
## Concatenating at side
#print(pd.concat( [df2,df3], axis = 1 ))


''' Reshaping & Pivot tables '''

#df = pd.DataFrame([
#            ['PID0', 'Gold', '1$', '2£'],
#            ['PID0', 'Bronze', '2$', '4£'],
#            ['PID1', 'Gold', '3$', '6£'],
#            ['PID1', 'Silver', '4$', '8£'],
#            ], columns = ['PID', 'CType', 
#                'USD', 'PND'])
#                  
#print(df)    
#print(df.pivot(index = 'PID', columns = 'CType'))  
#print(df.pivot(index = 'PID', columns = 'CType',
#               values = 'USD'))    

#df = pd.DataFrame([
#            ['PID0', 'Gold', '1$'],
#            ['PID0', 'Bronze', '2$'],
#            ['PID0', 'Gold', '3$'],     #Duplicate
#            ['PID1', 'Silver', '4$'],
#            ], columns = ['PID', 'CType', 
#                'USD'])

## Ambiguity in saving value to PID0_Gold 
## on values 1$ and 3$
#
#print(df)
#print(df.pivot_table(index = 'PID', 
#                     columns = 'CType',
#                     aggfunc = np.max))
    


# RESHAPING

#multi_ind = pd.MultiIndex.from_tuples(
#                                      [('IND','R1'),
#                                       ('IND','R2'),
#                                       ('US','R1'),
#                                       ('US','R2')],
#                                       names = [
#                                                '1st',
#                                                '2nd'])
#df = pd.DataFrame(np.random.randn(4,2),
#                  index = multi_ind,
#                  columns = ['C1', 'C2'])

#print(df)
#print(df.stack())   # increase in length
#print(df.unstack()) # increase in width



''' Grouping '''

#df = pd.DataFrame([["B", 1],
#                   ["B", 2],
#                   ["A", 3],
#                   ["A", 4]], columns = list('XY'))
#
#print(df)
#print(df.groupby(['X']).sum())
#print(df.groupby(['X'], sort = False).sum())


#df = pd.DataFrame({'A' : list('ppppqqqq'),
#                   'B' : list('rrssrrss'),
#                    'C': np.random.randn(8),
#                    'D': np.random.randn(8)})
#
#print(df)
#print(df.groupby(['A', 'B']).agg(
#                 {'C': {'C_mean': 'mean'},
#                  'D': {'D_median': 'median'}
#                   }))



''' Import and Export '''

#df = pd.DataFrame([[11, 202],
#                   [33, 44]],
#                   index = list('AB'),
#                    columns = list('CD'))
#
## Writing to excel file
#df.to_excel('files/pd_df.xlsx', sheet_name = 'Sheet1')
#
## Reading from excel file
#print(pd.read_excel('files/pd_df.xlsx', 'Sheet1'))