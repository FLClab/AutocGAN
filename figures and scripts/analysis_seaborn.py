import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Run this script to generate figure 3
# from Neural Architecture Optimization with Beam 
# Search for Conditional Generative Adversarial Networks 

archs = pd.read_csv('archs.csv')

for idx, param in enumerate(['C0', 'N0', 'U0', 'S0', 'C1', 'N1', 'U1', 'S1', 'K1', 'C2', 'N2', 'U2', 'S2', 'K2']):
	sns.jointplot(data=archs, x="IS", y="FID", hue=param, palette="tab10", height=5, ratio=4)
	plt.savefig('{}.pdf'.format(param))

plt.show()
