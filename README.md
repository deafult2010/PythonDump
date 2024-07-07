# PythonDump

# Step 0: Imports
import itertools
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sys import getsizeof
import math


# Step 1: Read data from each CSV
stats = pd.read_csv('teamStats.csv')
picks = pd.read_csv('playerPicks.csv')


# Step 2: Create a 24 x 41 array of each person's 4 picks. Each column represents a person's picks with points for picks else 0s.
# To do this:
# First Create new empty DF with column name headers
result_df = picks.iloc[:0,:].copy()
# Then iterate through the stats rows
for index, row in stats.iterrows():
    temp_dict = {}
    # checking for each person whether they have picked the current stats row team
    for col in result_df.columns:
        # if yes then set output the points for that team
        if row['Team'] in picks[col].values:
            temp_dict[col] = row['Value']
        # else output 0
        else:
            temp_dict[col] = 0
    # append the result to the newly created dataframe for each team thus adding 24 rows with 41 columns
    result_df = result_df._append(temp_dict, ignore_index=True)


# Step 3: Create a 735,471 x 24 binary array to establish all qualifying outcomes (in our example 1s represent a loss).
combinations = list(itertools.product([0,1], repeat=24))
filtCom = [row for row in combinations if sum(row)==8]
array = np.array(filtCom)


# Step 4: Assign the qualifying probabilities to the binary array. Note PR(!Qualify) = 1- Pr(Qualify)
# To do this:
# We'll first need to transpose the qualifying probabilities
probArray = stats.iloc[:,-1:].values.transpose()
# Then we'll need to assign the probabilities to the binary array
TJArray = np.abs(array-probArray)


# Step 5: Determine the probability of each outcome by computing the product of each row
prodArray = np.prod(TJArray, axis=1, keepdims=True)


# Step 6: Compute a factor to apply to the probabilities to ensure the sum f all possible outcomes sums to one*
# *In our case we are using Ladbrokes' qualifying odds which sum to more than one as they include a spread to ensure they make money
prodNum = np.sum(prodArray)


# Step 7: Adjust the prodArray by prodNum
prodArray2 = prodArray/prodNum
# Check that the sum of the probabilities sums to ones as intended. 
print(np.sum(prodArray2))


# Step 8: Determine each person's score under each outcome by multiplying the array & result_df arrays.
# We'll first need to convert the elements of both arrays to uint16 to save memory space. An alternative (more complex) approach could be to compute this in chunks.
# int64 by default is 8x larger in memory than is needed (uint is positive values only).
array =np.array(array, dtype=np.uint16)
result_df =np.array(result_df, dtype=np.uint16)
dotArray = np.dot(np.where(array==0,1,0),result_df)
# Debug array size
size = getsizeof(dotArray)/(1024.0**2)
print('df size: %2.2f MB'%size)


# Step 9: Rank the people based on their scores under each outcome
# Here we make use of the Scipy library to rank the array
rnkArray = len(dotArray[0]) - rankdata(dotArray,axis=1)


# Step 10: multiply the rnkArray by the prodArray2
probRnkArray = rnkArray * prodArray2


# Step 11: Output the expected position and standard deviation of peoples finishing positions
posArray = np.sum(probRnkArray, axis=0, keepdims=True)
stDevArray = np.std(probRnkArray, axis=0, keepdims=True)*math.comb(24, 16)/24
# Using a similar approach to step 2 we create a new data frame and append rows to it
results_df = picks.iloc[:0,:].copy()
results_df = results_df._append(pd.DataFrame(posArray, columns=results_df.columns), ignore_index=True)
results_df = results_df._append(pd.DataFrame(stDevArray, columns=results_df.columns), ignore_index=True)
# Then we output this dataframe to CSV
results_df.to_csv('output.csv', index=False)


# Step 12 (Optional): Output the probability density of peoples finishing position
rnk_df = picks.iloc[:0,:].copy()
rnk_df = rnk_df._append(pd.DataFrame(rnkArray, columns=rnk_df.columns), ignore_index=True)
prodArrayDf = pd.DataFrame(prodArray2, columns=['Prob'])
rnk_df = pd.concat([rnk_df, prodArrayDf], axis=1)
output_df = picks.iloc[:0,:].copy()
output_df['Bucket'] = np.arange(0.0, 41.0, 0.5)
for col in picks.iloc[:0,:]:
    bucket_prob = rnk_df.groupby(col)['Prob'].sum()
    output_df[col] = [bucket_prob.get(i, 0) for i in np.arange(0.0, 41.0, 0.5)]
output_df.to_csv('output3.csv', index=False)
