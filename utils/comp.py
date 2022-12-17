import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""

Choose matrix OR model to compare

"""

df = pd.read_pickle('matrix_comp.pkl')
fig = plt.figure(figsize=(15,12))
colors=['C9E6BB', '9FCCD0', 'C3D065','6BB49D'] #matrix
# colors = ['F2EFD5', 'C7DBE7', 'C7D2F2', '9FA6D6'] #model
plt.gca().set_prop_cycle(color=colors)
for i in df.index:
    plt.plot(df.loc[i]['fpr'], 
             df.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, df.loc[i]['auc']), linewidth = 3)

plt.plot([0,1], [0,1], color='grey', linestyle='--',linewidth = 3)
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

#plt.show()
fig.savefig('matrix_comp.png')