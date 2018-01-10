# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../datasets/Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

avg_rwds = [0] * d
deltas = [0] * d
  
all_avgs = np.zeros((N,d))
all_deltas = np.zeros((N,d))
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = float(sums_of_rewards[i]) / numbers_of_selections[i]
            delta_i = math.sqrt((3.0/2.0) * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
            avg_rwds[i] = average_reward
            deltas[i] = delta_i
            all_avgs[n,i] = average_reward
            all_deltas[n,i] = delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    """if (n+1) % 1000 == 0:
        plt.errorbar(range(d),avg_rwds,yerr=deltas,xerr=0.3,ls = 'None', marker = '_')
        plt.show()    
        plt.hist(ads_selected)
        plt.title('Histogram of ads selections')
        plt.xlabel('Ads')
        plt.ylabel('Number of times each ad was selected')
        plt.show()"""
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

"""plt.plot(range(N), all_avgs[:,0])# + all_deltas[:, 0])
plt.plot(range(N), all_avgs[:,1])# + all_deltas[:, 1])#,yerr=all_deltas[:,1])
plt.plot(range(N), all_avgs[:,2])# + all_deltas[:, 2])#,yerr=all_deltas[:,2])
plt.plot(range(N), all_avgs[:,3])# + all_deltas[:, 3])#,yerr=all_deltas[:,3])
plt.plot(range(N), all_avgs[:,4])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.plot(range(N), all_avgs[:,5])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.plot(range(N), all_avgs[:,6])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.plot(range(N), all_avgs[:,7])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.plot(range(N), all_avgs[:,8])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.plot(range(N), all_avgs[:,9])# + all_deltas[:, 4])#,yerr=all_deltas[:,4])
plt.legend(['0','1','2','3','4','5','6','7','8','9'])
plt.show()"""
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()