import pandas as pd
import numpy as np
from utils.data_preprocessor import useful_genes


class CancerDataset():
    """
    feature_transformer is of type FeatureTransformer
    """
    
    def __init__(self, file_name="skcm_dataset.csv", matrix_type = 'ALL',selection = True):
        
        if ".csv" in file_name:
            df =  pd.read_csv(file_name, index_col=0)
            #initiate the useful_genes class
            gene_obj = useful_genes()
            genes = gene_obj.get_genes(matrix_type = matrix_type)
            drop_feature_list = gene_obj.get_drop()
            #make sure useful gene exist
            genes = list(set(df.columns.tolist()) & set(genes))
            
            self.data = df[genes].to_numpy() if selection else df.drop(drop_feature_list, axis = 1).to_numpy()
            
            self.label = df['OS'].tolist()
            self.feature = genes
            self.shape = (len(self.data), len(self.data[0]))
            
            
        else:
            # TODO: include data transformation methods
            raise Exception("bad file naming, check your file name again to ensure it is a csv file")
            
            
    def __len__(self):
        return len(self.y_train)
            
    def __shape__(self):
        return self.shape
    
    def __getitem__(self,index):
        gene = self.data[index]
        label = self.label[index]
        sample = gene,label
        # if self.transform:
        #     sample = self.transform(sample)
        return sample

    
class Cox_CancerDataset():
    """
    feature_transformer is of type FeatureTransformer
    """
    
    def __init__(self, file_name="skcm_dataset.csv", matrix_type = 'ALL',selection = True):
        
        if ".csv" in file_name:
            df =  pd.read_csv(file_name, index_col=0)
            gene_obj = useful_genes()
            genes = gene_obj.get_genes(matrix_type = matrix_type)
            drop_feature_list = gene_obj.get_drop()
            #make sure useful gene exist
            useful_info = list(set(df.columns.tolist()) & set(genes))
            self.data = df[useful_info].to_numpy() if selection else df.drop(drop_feature_list, axis = 1).to_numpy()
    
            self.label = df['OS.time'].tolist()
            self.feature = useful_info
            self.shape = (len(self.data), len(self.data[0]))
            useful_info.extend(['OS.time','OS'])
            self.matrix = df[useful_info]


            
        else:
            # TODO: include data transformation methods
            raise Exception("bad file naming, check your file name again to ensure it is a csv file")
            
            
    def __len__(self):
        return len(self.y_train)
            
    def __shape__(self):
        return self.shape
    
    def __getitem__(self,index):
        gene = self.data[index]
        label = self.label[index]
        sample = gene,label
        # if self.transform:
        #     sample = self.transform(sample)
        return sample
    def __matrix__(self):
        
        return self.matrix
    
    
    # Use the following default feature transformer when necessary
class FeatureTransformer:
    def __init__(self):
        super().__init__()

    @staticmethod
    def fit_transform(train_X):
        
        return train_X