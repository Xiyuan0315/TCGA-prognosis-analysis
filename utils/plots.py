import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import pandas as pd

"""
This section provides plots for 
1. Dataset distribution
2. Univariable cox regression with covariates
...
"""
summary_col = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
       'exp(coef) lower 95%', 'exp(coef) upper 95%', 'cmp to', 'z', 'p',
       '-log2(p)']
dur_eve = ['week','arrest']



def plot_barh(tcga):

    D = [('SKCM',np.count_nonzero(tcga.label),len(tcga.label)-np.count_nonzero(tcga.label))] #enter data for language & usage, 3rd column is for 2017 usage

    cancer_type = [x[0] for x in D] #create a list from the first dimension of data
    death  = [x[1] for x in D] #create a list from the second dimension of data (2018 popularity)
    alive  = [x[2] for x in D] #create a list from the second dimension of data (2017 popularity)

    ind = np.arange(len(cancer_type))
    width=0.3 

    fig,ax = plt.subplots(figsize=(4, 2))
    ax.barh(ind, death, width, align='center', alpha=0.8, color='#c1cbd7', label='Death') #a horizontal bar chart (use .bar instead of .barh for vertical)
    ax.barh(ind - width, alive, width, align='center', alpha=0.8, color='#add2a3', label='Alive') #a horizontal bar chart (use .bar instead of .barh for vertical)
    ax.set(yticks=ind - width/2, yticklabels=cancer_type, ylim=[width - 1, len(cancer_type)])

    for i, v in enumerate(death):
        ax.text(v+0.15,i-0.05, str(v), fontsize=8) #the 0.15 and 0.05 were set after trial & error (based on how nice things look)
    for i, v in enumerate(alive):
        ax.text(v+0.15,i-0.4, str(v), fontsize=8) #the 0.4 was set after trial & error (based on how nicely it aligns, edit it and rerun to see the difference)

    plt.xlabel('Numbers of patients', fontsize = 5)
    plt.title('Distribution of SKCM dataset from TCGA', fontsize = 9)
    plt.legend(fontsize = 7)
    return fig

def get_summary(df,covar, var):
    # Get the hear of summary matrix
    global summary_col, dur_eve
    final_tb  = pd.DataFrame(columns=summary_col)
    for i in range(len(var)):
        covar.append(var[i])
        final_col = covar + dur_eve
        sub_df = df[final_col]
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(sub_df,dur_eve[0], dur_eve[1])
        final_tb.loc[var[i]] = cph.summary.loc[var[i]].tolist()

    return final_tb.sort_values(by='coef', ascending=False)

# 3. Start ploting

def plot_cox(df,ax=None, hazard_ratios = False,**errorbar_kwargs):

    if ax is None:
        fig,ax = plt.subplots()

    errorbar_kwargs.setdefault("c", "k")
    errorbar_kwargs.setdefault("fmt", "s")
    errorbar_kwargs.setdefault("markerfacecolor", "white")
    errorbar_kwargs.setdefault("markeredgewidth", 1.25)
    errorbar_kwargs.setdefault("elinewidth", 1.25)
    errorbar_kwargs.setdefault("capsize", 3)

    columns = df.index
    yaxis_locations = list(range(len(columns)))
    
    if hazard_ratios: #exp(coef)
        exp_log_hazards = df['exp(coef)'].tolist() 
        errors = np.subtract(df['exp(coef) upper 95%'].tolist(),df['exp(coef)'].tolist())
        ax.errorbar(
            exp_log_hazards,
            yaxis_locations,
            xerr=errors,
            **errorbar_kwargs)
        ax.set_xlabel("HR (%g%% CI)" % ((1) * 100))

    else: #coef
        log_hazards = df['coef'].tolist() 
        errors = np.subtract(df['coef upper 95%'].tolist(),df['coef'].tolist()) 
        ax.errorbar(log_hazards, yaxis_locations, 
            xerr=errors, 
            **errorbar_kwargs)
        ax.set_xlabel("log(HR) (%g%% CI)" % ((1) * 100))



    best_ylim = ax.get_ylim()
    ax.vlines(1 if hazard_ratios else 0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65, color="k")
    ax.set_ylim(best_ylim)

    tick_labels = columns

    ax.set_yticks(yaxis_locations)
    ax.set_yticklabels(tick_labels)

    return fig

