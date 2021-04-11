import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from surprise import SVD
from surprise import KNNBasic
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

######################
# Testing Parameters #
######################

cpus = 1

#####################
# Pre process data  #
#####################

df = pd.read_csv('ratings_small.csv')
df = df[['userId', 'movieId', 'rating']]

# Create a Reader for ratings.csv
ratings_reader = Reader()

data = Dataset.load_from_df(df, ratings_reader)

item_rmse = []
item_mae = []

user_rmse = []
user_mae = []

####################################################
# Calculate PMF, User CF, Item CF (MSD similarity) #
####################################################

print("MSD Similarity")
print("==============")

# PMF
algo = SVD(verbose=False)
# Run 5-fold cross-validation and print results.
print("Probabilistic Matrix Factorization")
svd_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", svd_results.get('test_rmse').mean())
print("Mean MAE:\t", svd_results.get('test_mae').mean(), "\n")

# User based collaborative filtering
sim = {'name': 'MSD', 'user_based': True}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("User-based CF")
user_cf_results_msd = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", user_cf_results_msd.get('test_rmse').mean())
print("Mean MAE:\t", user_cf_results_msd.get('test_mae').mean(), "\n")

# Append results to array
user_rmse.append(user_cf_results_msd.get('test_rmse').mean())
user_mae.append(user_cf_results_msd.get('test_mae').mean())

# Item based collaborative filtering
sim = {'name': 'MSD', 'user_based': False}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("Item-based CF")
item_cf_results_msd = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", item_cf_results_msd.get('test_rmse').mean())
print("Mean MAE:\t", item_cf_results_msd.get('test_mae').mean(), "\n\n")

# Append results to array
item_rmse.append(item_cf_results_msd.get('test_rmse').mean())
item_mae.append(item_cf_results_msd.get('test_mae').mean())

# ##################################################
# # Calculate User CF, Item CF (Cosine similarity) #
# ##################################################

print("Cosine Similarity")
print("=================")

# User based collaborative filtering
sim = {'name': 'cosine', 'user_based': True}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("User-based CF")
user_cf_results_cosine = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", user_cf_results_cosine.get('test_rmse').mean())
print("Mean MAE:\t", user_cf_results_cosine.get('test_mae').mean(), "\n")

# Append results to array
user_rmse.append(user_cf_results_cosine.get('test_rmse').mean())
user_mae.append(user_cf_results_cosine.get('test_mae').mean())

# Item based collaborative filtering
sim = {'name': 'cosine', 'user_based': False}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("Item-based CF")
item_cf_results_cosine = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", item_cf_results_cosine.get('test_rmse').mean())
print("Mean MAE:\t", item_cf_results_cosine.get('test_mae').mean(), "\n\n")

# Append results to array
item_rmse.append(item_cf_results_cosine.get('test_rmse').mean())
item_mae.append(item_cf_results_cosine.get('test_mae').mean())

###################################################
# Calculate User CF, Item CF (Pearson similarity) #
###################################################

print("Pearson Similarity")
print("==================")

# User based collaborative filtering
sim = {'name': 'pearson', 'user_based': True}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("User-based CF")
user_cf_results_pearson = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", user_cf_results_pearson.get('test_rmse').mean())
print("Mean MAE:\t", user_cf_results_pearson.get('test_mae').mean(), "\n")

# Append results to array
user_rmse.append(user_cf_results_pearson.get('test_rmse').mean())
user_mae.append(user_cf_results_pearson.get('test_mae').mean())

# Item based collaborative filtering
sim = {'name': 'pearson', 'user_based': False}
algo = KNNBasic(sim_options=sim, verbose=False)

# Run 5-fold cross-validation and print results.
print("Item-based CF")
item_cf_results_pearson = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
print("Mean RMSE:\t", item_cf_results_pearson.get('test_rmse').mean())
print("Mean MAE:\t", item_cf_results_pearson.get('test_mae').mean(), "\n\n")

item_rmse.append(item_cf_results_pearson.get('test_rmse').mean())
item_mae.append(item_cf_results_pearson.get('test_mae').mean())

#####################################################################
# Calculate User CF, Item CF (MSD similarity) varying k from 1 - 40 #
#####################################################################

# Run from K = 1 to K
maxK = 40

user_rmse_k = []
user_mae_k = []

item_rmse_k = []
item_mae_k = []

for k in range(1, maxK+1):
    print("Running User CF with k =", k)
    # User based collaborative filtering
    sim = {'name': 'MSD', 'user_based': True}
    algo = KNNBasic(k=k, sim_options=sim, verbose=False)

    # Run 5-fold cross-validation and print results.
    print("User-based CF")
    user = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
    print("Mean RMSE:\t", user.get('test_rmse').mean())
    print("Mean MAE:\t", user.get('test_mae').mean(), "\n")

    # Append results to array
    user_rmse_k.append(user.get('test_rmse').mean())
    user_mae_k.append(user.get('test_mae').mean())

    # Item based collaborative filtering
    sim = {'name': 'MSD', 'user_based': False}
    algo = KNNBasic(k=k, sim_options=sim, verbose=False)

    print("Running Item CF with k =", k)
    # Run 5-fold cross-validation and print results.
    print("Item-based CF")
    item = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=False, n_jobs=cpus)
    print("Mean RMSE:\t", item.get('test_rmse').mean())
    print("Mean MAE:\t", item.get('test_mae').mean(), "\n\n")

    # Append results to array
    item_rmse_k.append(item.get('test_rmse').mean())
    item_mae_k.append(item.get('test_mae').mean())


print("Minimum User RMSE:",min(user_rmse_k),"at k =",user_rmse_k.index(min(user_rmse_k)))
print("Minimum User MAE:",min(user_mae_k),"at k =",user_mae_k.index(min(user_mae_k)))
print("Minimum Item RMSE:",min(item_rmse_k),"at k =",item_rmse_k.index(min(item_rmse_k)))
print("Minimum Item MAE:",min(item_mae_k),"at k =",item_mae_k.index(min(item_mae_k)))

################
# Plot Results #
################

labels = ['MSD', 'Cosine', 'Pearson']

x = np.arange(len(labels)) # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - 6*width/4, user_rmse, width, label='User-based (RMSE)', color='orangered')
rects2 = ax.bar(x - 2*width/4, item_rmse, width, label='Item-based (RMSE)', color='orange')
rects3 = ax.bar(x + 2*width/4, user_mae,  width, label='User-based (MAE)', color='teal')
rects4 = ax.bar(x + 6*width/4, item_mae,  width, label='Item-based (MAE)', color='turquoise')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Comparison of Item and User based CF Error')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

for bar in rects1:
    height = bar.get_height()
    label_x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(label_x_pos, height, s=f'{round(height, 3)}', ha='center', va='bottom')

for bar in rects2:
    height = bar.get_height()
    label_x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(label_x_pos, height, s=f'{round(height, 3)}', ha='center', va='bottom')

for bar in rects3:
    height = bar.get_height()
    label_x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(label_x_pos, height, s=f'{round(height, 3)}', ha='center', va='bottom')

for bar in rects4:
    height = bar.get_height()
    label_x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(label_x_pos, height, s=f'{round(height, 3)}', ha='center', va='bottom')


ax.legend(bbox_to_anchor=(1,1), loc="upper left")
fig.tight_layout()

plt.savefig('user_item_rmse.png')
plt.show()

##########################################################################
# Part f

x = np.arange(1, maxK+1)

fig, ax = plt.subplots()
plt.plot(x, user_rmse_k, label='User-based (RMSE)')
plt.plot(x, user_mae_k,  label='User-based (MAE)')
plt.plot(x, item_rmse_k, label='Item-based (RMSE)')
plt.plot(x, item_mae_k, label='Item-based (MAE)')


ax.legend()
ax.xaxis.grid(True, which='both')
ax.set_xlabel('K')
ax.set_ylabel('Error')
ax.set_title('Item-based and User-based CF Error over varying K values')

# Set ticks
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))

plt.tight_layout()

plt.savefig('./different_k_values', format='png')
plt.show()

