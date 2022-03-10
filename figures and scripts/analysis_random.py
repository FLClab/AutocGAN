import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib
import random
import seaborn as sns

#CONV_TYPE = {0: "post", 1: "pre"}
#NORM_TYPE = {0: None, 1: "bn", 2: "in"}
#UP_TYPE = {0: "bilinear", 1: "nearest", 2: "deconv"}
#SHORT_CUT_TYPE = {0: False, 1: True}
#SKIP_TYPE = {0: False, 1: True}

font = {'size'   : 16}

matplotlib.rc('font', **font)

archs = pd.read_csv('archs_random.csv')

IS = np.array(archs['IS'])
FID = np.array(archs['FID'])


# Computing time to reach better IS than Greedy
is_to_beat = 8.27
iterations_for_better = []
for i in range(100000):
	random = 0
	i = 0
	while random < is_to_beat:
		random = np.random.choice(IS, 1)
		i += 1
	iterations_for_better.append(i*10) # GPU-hours

plt.hist(iterations_for_better, bins=np.arange(0,np.max(iterations_for_better),np.max(iterations_for_better)/20), edgecolor='k', color='#1f76b4ff', alpha=0.5)
plt.xlabel('GPU hours to beat IS={}'.format(is_to_beat))
plt.ylabel('# trials out of 100,000')
print(np.mean(iterations_for_better))
plt.tight_layout()
plt.show()

if 0:
	# FID/IS plot: random vs. searched
	archs_searched = pd.read_csv('archs.csv')

	fig, ax = plt.subplots(1,1)

	IS_searched = np.array(archs_searched['IS'])
	FID_searched = np.array(archs_searched['FID'])

	axs = sns.jointplot('IS', 'FID', data=archs_searched)
	axs.ax_joint.scatter('IS', 'FID', data=archs, c='r', marker='o', edgecolor='w')

	# drawing pdf instead of histograms on the marginal axes
	axs.ax_marg_x.cla()
	axs.ax_marg_y.cla()
	sns.distplot(archs_searched.IS, ax=axs.ax_marg_x)
	sns.distplot(archs_searched.FID, ax=axs.ax_marg_y, vertical=True)

	# Adding hist and pdf for random
	ax = sns.distplot(archs.IS, ax=axs.ax_marg_x, color='r')
	ax = sns.distplot(archs.FID, ax=axs.ax_marg_y, color='r', vertical=True)

	plt.tight_layout()
	plt.show()