# PythonDump

import itertools
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sys import getsizeof
import math


stats = pd.read_csv('teamStats.csv')
print(stats.to_string())

picks = pd.read_csv('playerPicks.csv')
print(picks.to_string())

probArray = stats.iloc[:,-1:].values.transpose()
print(probArray)

#Create new empty DF with column name headers
result_df = picks.iloc[:0,:].copy()

for index, row in stats.iterrows():
    temp_dict = {}
    for col in result_df.columns:
        if row['Team'] in picks[col].values:
            temp_dict[col] = row['Value']
        else:
            temp_dict[col] = 0
    result_df = result_df._append(temp_dict, ignore_index=True)

print(result_df)

combinations = list(itertools.product([0,1], repeat=24))

filtCom = [row for row in combinations if sum(row)==8]
array = np.array(filtCom)
print(array)

TJArray = np.abs(1-np.abs(array-probArray))
print(TJArray)

prodArray = np.prod(TJArray, axis=1, keepdims=True)
print(prodArray)

prodNum = np.sum(prodArray)
print(prodNum)

prodArray2 = prodArray/prodNum
print(prodArray2)

#CHECK
print(np.sum(prodArray2))

#int64 by default is 4x larger in memory than is needed.
array =np.array(array, dtype=np.uint16)
result_df =np.array(result_df, dtype=np.uint16)

dotArray = np.dot(array,result_df)
print(dotArray)

#Debug array size
size = getsizeof(dotArray)/(1024.0**2)
print('df size: %2.2f MB'%size)

# rnkArray = rankdata(dotArray,axis=1)
# reverse rank
rnkArray = rankdata([-1*i for i in dotArray],axis=1)
print(rnkArray)

probRnkArray = rnkArray * prodArray2
print(probRnkArray)

posArray = np.sum(probRnkArray, axis=0, keepdims=True)
print(posArray)

stDevArray = np.std(probRnkArray, axis=0, keepdims=True)*math.comb(24, 16)/24
print(stDevArray)


results_df = picks.iloc[:0,:].copy()
results_df = results_df._append(pd.DataFrame(posArray, columns=results_df.columns), ignore_index=True)
results_df = results_df._append(pd.DataFrame(stDevArray, columns=results_df.columns), ignore_index=True)
print(results_df)


results_df.to_csv('output.csv', index=False)

rnk_df = picks.iloc[:0,:].copy()
rnk_df = rnk_df._append(pd.DataFrame(rnkArray, columns=rnk_df.columns), ignore_index=True)
prodArrayDf = pd.DataFrame(prodArray2, columns=['Prob'])
rnk_df = pd.concat([rnk_df, prodArrayDf], axis=1)
print(rnk_df)

output_df = picks.iloc[:0,:].copy()
output_df['Bucket'] = np.arange(0.0, 41.0, 0.5)
print(output_df)

for col in picks.iloc[:0,:]:
    bucket_prob = rnk_df.groupby(col)['Prob'].sum()
    output_df[col] = [bucket_prob.get(i, 0) for i in np.arange(0.0, 41.0, 0.5)]
print(output_df)

output_df.to_csv('output3.csv', index=False)
