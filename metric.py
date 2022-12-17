import matplotlib.pyplot as plt
import pandas as pd
import statistics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn import metrics
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
def acc(y_test, y_pred):
    
    return metrics.accuracy_score(y_test, y_pred)
    
def sens_spec(y_test, y_pred):
    
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    matrix = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=[0,1])

        
    return [tp/(tp + fn),tn/(fp + tn)], matrix
    

def roc_auc(y_test, y_pred):
    fig,ax = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    RocCurveDisplay.from_predictions(y_test, y_pred).plot(ax = ax,color = '#C9E8BB',linestyle = '--', linewidth = 2)
    
    # ax.plot([0,1], [0,1], color='grey', linestyle='--',linewidth = 3) 

    return list(fpr), list(tpr), auc,fig
    
    
def get_feature_rank(clf, X_test, y_test, gene_list):
# permutation_importance
    result = permutation_importance(
        clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    
    """
    permuration_importance: the decrease in a model score 
    when this single feature value is randomly shuffled
    """
    forest_importances = pd.Series(result.importances_mean, index = gene_list)
    forest_importances = forest_importances.sort_values(ascending = False) #plot in desceding order
    
    fig, ax = plt.subplots(figsize=(15,12))

    forest_importances.plot.bar(yerr=result.importances_std,
    ax=ax, color = '#C9E8BB')
    ax.set_title("Feature importances using permutation on full model", fontsize=25)
    ax.set_ylabel("Mean accuracy decrease",fontsize=18 )
    fig.tight_layout()
    return forest_importances, fig 


def get_sep_gene(df,gene):
    df = df[[gene,'OS','OS.time']]
    
    median = statistics.median(df[gene].tolist())
    df_low = df.loc[df[gene] < median]
    df_high = df.loc[df[gene] > median]
    
    return df_low, df_high


def get_p(df_low, df_high):
    result = logrank_test(df_low['OS.time'], df_high['OS.time'], event_observed_A=df_low['OS'], event_observed_B=df_high['OS'])
    return result.p_value

def plot_surv(df,gene):
    fig,ax = plt.subplots()

    df_low, df_high = get_sep_gene(df,gene)
    p_val = get_p(df_low, df_high)
    kmf = KaplanMeierFitter(label = 'skcm dataset')
    kmf.fit(durations = df_low['OS.time'], event_observed=df_low['OS'], label="Low")
    kmf.plot_survival_function(ax=ax, color = '#F1C860')

    kmf.fit(durations = df_high['OS.time'], event_observed=df_high['OS'],label="High")
    kmf.plot_survival_function(ax=ax, color = '#6BB49D')
    
    plt.xlabel('Month')
    plt.title(f"Survival Analysis on {gene}")
    style = dict(size=10, color='black', weight='bold')
    ax.text(62,0.8,f'p value: {round(p_val,4)}',ha = 'right', **style)
    return fig

