import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib

# Run this script to generate suppl. figure ? 
# from Neural Architecture Optimization with Beam 
# Search for Conditional Generative Adversarial Networks 

font = {'size'   : 16}

matplotlib.rc('font', **font)

archs = pd.read_csv('archs_best.csv')

fig, ax = plt.subplots(1,1)

z0 = 300
z1 = 3*z0
z2 = 6*z0

### Block2
C2 = np.array(archs['C2'])
N2 = np.array(archs['N2'])
U2 = np.array(archs['U2'])
S2 = np.array(archs['S2'])
K2 = np.array(archs['K2'])

### Block1
C1 = np.array(archs['C1'])
N1 = np.array(archs['N1'])
U1 = np.array(archs['U1'])
S1 = np.array(archs['S1'])
K1 = np.array(archs['K1'])

### Block0
C0 = np.array(archs['C0'])
N0 = np.array(archs['N0'])
U0 = np.array(archs['U0'])
S0 = np.array(archs['S0'])

IS = np.array(archs['IS'])
FID = np.array(archs['FID'])


markers = ['P', 'X']   # Pre, post
colors = ['w','k']#['#fbd403','#d60023'] # None, BN, IN
edgecolors = ['k', '#30757e', '#002f6c','#754c28']
facecolors = ['#1b4169','#fbd403','#d60023'] # bilinear, nearest, deconv

for c0,c1,c2,n0,n1,n2,u0,u1,k1,u2,s0,s1,s2,k2,FID_,IS_ in zip(C0,C1,C2,N0,N1,N2,U0,U1,K1,U2,S0,S1,S2,K2,FID,IS):

	ax.scatter(FID_, IS_, s=z2, marker='o', facecolor=facecolors[u2], edgecolor=edgecolors[k2], linewidth=2)
	ax.scatter(FID_, IS_, c=colors[n2], s=z2, marker=markers[c2], edgecolor='k', linewidth=1)

	ax.scatter(FID_, IS_, s=z1, marker='o', facecolor=facecolors[u1], edgecolor=edgecolors[k1], linewidth=2)
	ax.scatter(FID_, IS_, c=colors[n1], s=z1, marker=markers[c1], edgecolor='k', linewidth=1)

	ax.scatter(FID_, IS_, s=z0, marker='o', facecolor=facecolors[u0], edgecolor='k', linewidth=1)
	ax.scatter(FID_, IS_, c=colors[n0], s=z0, marker=markers[c0], edgecolor='k', linewidth=1)


# Custom legend
from matplotlib.lines import Line2D
upsampling = [Line2D([0], [0], marker='o', color='w', markerfacecolor=facecolors[0], markersize=15, markeredgecolor='k'),
			  Line2D([0], [0], marker='o', color='w', markerfacecolor=facecolors[1], markersize=15, markeredgecolor='k'),
			  Line2D([0], [0], marker='o', color='w', markerfacecolor=facecolors[2], markersize=15, markeredgecolor='k')]

normalization = [Line2D([0], [0], marker='P', color='w', markerfacecolor=colors[0], markersize=15, markeredgecolor='k'),
				 Line2D([0], [0], marker='P', color='w', markerfacecolor=colors[1], markersize=15, markeredgecolor='k')]

convolution = [Line2D([0], [0], marker='P', color='w', markerfacecolor='w', markeredgecolor='k', markersize=15),
			   Line2D([0], [0], marker='X', color='w', markerfacecolor='w', markeredgecolor='k', markersize=15)]

skiplinks = [Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor=edgecolors[0], markersize=10, markeredgewidth=3),
			 Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor=edgecolors[1], markersize=10, markeredgewidth=3),
			 Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor=edgecolors[2], markersize=10, markeredgewidth=3),
			 Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor=edgecolors[3], markersize=10, markeredgewidth=3)]

stage = [Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='k', markersize=10),
		 Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='k', markersize=20),
		 Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markeredgecolor='k', markersize=30)]

legend1 = plt.legend(upsampling, ['Bilinear', 'Nearest', 'Deconv'], loc='lower right', bbox_to_anchor=(0.3, 0.15))
legend2 = plt.legend(normalization, ['None', 'Batch'], loc='lower left', bbox_to_anchor=(0.3, 0.15))
legend3 = plt.legend(convolution, ['Pre', 'Post'], loc='upper left', bbox_to_anchor=(0.3, 0.15))
legend4 = plt.legend(stage, ['Block0', 'Block1', 'Block2'], loc='upper right', bbox_to_anchor=(0.3, 0.15))
legend5 = plt.legend(skiplinks, ['None', 'With 0', 'With 1', 'With 0 & 1'], loc='upper right', bbox_to_anchor=(0.3, 0.5))

ax.add_artist(legend1)
ax.add_artist(legend2)
ax.add_artist(legend3)
ax.add_artist(legend4)
ax.add_artist(legend5)

ax.set_xlabel('FID')
ax.set_ylabel('Inception Score')
plt.tight_layout()
plt.show()
