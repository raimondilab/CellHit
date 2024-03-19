import faiss
import pandas as pd
import numpy as np

class FaissKNN:
    def __init__(self, df):
        self.df = df
        self.index = None
        self.index_names = None
        self.build_faiss_index()
        
    def build_faiss_index(self):
        # Converting DataFrame to numpy array
        
        self.index_names = list(self.df.index)
        feature_matrix = np.ascontiguousarray(self.df.to_numpy().astype('float32'))
        
        # Initializing and building the index
        self.index = faiss.IndexFlatL2(feature_matrix.shape[1])
        self.index.add(feature_matrix)
        
    def knn(self, query_df, k, skip_self=False):
        
        query_names = list(query_df.index)
        query_matrix = np.ascontiguousarray(query_df.to_numpy().astype('float32'))
        
        # Search for k+1 neighbors if we want to skip the query point itself
        k_search = k + 1 if skip_self else k
        
        # Conduct the search
        distances, indices = self.index.search(query_matrix, k_search)
        
        #print('prova')
        
        # Prepare the output DataFrame
        output_data = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for j, (dist, idx) in enumerate(zip(dist_row, idx_row)):
                # Skip the query point if needed
                if skip_self and i == idx:
                    continue
                output_data.append({
                    'query_point': query_names[i],
                    'neighbour_point': self.index_names[idx],
                    'ranking': j,
                    'distance': dist
                })
                
        return pd.DataFrame(output_data)