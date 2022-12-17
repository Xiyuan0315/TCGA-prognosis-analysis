import numpy as np
import pandas as pd

# From 5k normlization??


class useful_genes():
    def __init__(self):
        self.drop_feature_list = ['X_PATIENT', 'OS.time', 'OS']
        self.path = '/Users/xiyuanzhang/Desktop/SZBL/scRNA/cox/Bulk/filtered_gene/'
        self.gene_dic = {}
        self.gene_dic['CD8Tex'] = pd.read_csv(self.path + 'bayes_CD8Tex.csv',
                                usecols = [0])['gene'].tolist()

        self.gene_dic['tumor'] = pd.read_csv(self.path + 'bayes_tumor.csv',
                                usecols = [0])['gene'].tolist()
        self.gene_dic['tcell'] = pd.read_csv(self.path + 'bayes_tcell.csv',
                                usecols = [0])['gene'].tolist()
        self.gene_dic['tpm'] = pd.read_csv(self.path + 'tpm.csv',
                                usecols = [0])['gene'].tolist()
        
    def get_genes(self, matrix_type):
        if matrix_type == 'ALL':
            return np.unique(sum(self.gene_dic.values(), []))
        else:
            return self.gene_dic[matrix_type]
    def get_drop(self):
        return self.drop_feature_list

    def _add(self):
        pass






        