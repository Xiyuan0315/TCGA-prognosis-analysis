import streamlit as st
from base_dataset import CancerDataset, Cox_CancerDataset
from utils.data_preprocessor import useful_genes
import utils.plots
import metric
from lifelines.datasets import load_rossi

# Load data
tcga = CancerDataset(file_name = f'data/skcm_tpm.csv',matrix_type='ALL')
cox_obj = Cox_CancerDataset(file_name = 'data/skcm_tumor.csv', matrix_type='ALL')
df = cox_obj.__matrix__()

# Display the explore page
def show_explore_page(tcga = tcga):
    count_matrices = ('tpm',
    'tumor',
    'tcell',
    'CD8Tex')
    st.title("Explore TCGA dataset for SKCM")
    st.markdown('This section provides the overview of the dataset,cox regression and survival analysis on promissing genes from tpm data and cell specific data purified by [bayes prism](https://github.com/Danko-Lab/BayesPrism)')

    # 1. Distribution
    show = st.checkbox("Show dataset distribution")
    if show:
        dist = utils.plots.plot_barh(tcga)
        st.pyplot(dist)
    # 2. Cox regression
    st.subheader('Cox regression')
    count_matrix = st.selectbox("Count matrix",count_matrices)
    df_test = load_rossi()
    
    options = st.multiselect(
    'Select covariants:',
    ['fin','age'])
    final_tb = utils.plots.get_summary(df_test,covar = options, var = ['race','mar'])
    cox_fig = utils.plots.plot_cox(final_tb, hazard_ratios=True)
    cox = st.button(f'Run cox regression for {count_matrix}')
    if cox:
        st.pyplot(cox_fig)


    # 3. Survival Analysis
    st.subheader('Survival Analysis')
    gene_list = useful_genes().get_genes(matrix_type=count_matrix) # gene lists
    selected_gene = st.selectbox("Survival analysis of featured genes", gene_list) # selected gene
    st.pyplot(metric.plot_surv(df,selected_gene))
    with st.expander(f"All featured genes from {count_matrix} matrix:"):
        st.write(gene_list)