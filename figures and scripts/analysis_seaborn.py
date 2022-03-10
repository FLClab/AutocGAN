import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

#CONV_TYPE = {0: "post", 1: "pre"}
#NORM_TYPE = {0: None, 1: "bn", 2: "in"}
#UP_TYPE = {0: "bilinear", 1: "nearest", 2: "deconv"}
#SHORT_CUT_TYPE = {0: False, 1: True}
#SKIP_TYPE = {0: False, 1: True}

font = {'size'   : 16}

matplotlib.rc('font', **font)

archs = pd.read_csv('archs.csv')

for idx, param in enumerate(['C0', 'N0', 'U0', 'S0', 'C1', 'N1', 'U1', 'S1', 'K1', 'C2', 'N2', 'U2', 'S2', 'K2']):
	sns.jointplot(data=archs, x="IS", y="FID", hue=param, palette="tab10", height=5, ratio=4)
	plt.savefig('all_archs/{}.pdf'.format(param))

