import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class ProfileClustering:
    def __init__(self,num_of_component, num_of_cluster) -> None:
        self.component = num_of_component
        self.cluster = num_of_cluster
    
    def differentiate(self, df_pca_kmeans, num_of_cluster, method = 1):
        """
        input: df_pca_kmeans = dataframe containing the input data and their clusterId, clusterId = which cluster it is in
        output: method 1 gives a list of similar features and other method gives a list of 5 most similar features and list 
                of 5 most different feature

        This function will take the dataframe and clusterId and obtain their standard deviation. The 5 lowest standard deviation feature will be used as the most similar feature and 
        the 5 highest standard deviation feature will be used as the most different feature. The 10 features used will also show their mean and variance and will be returned with a list
        for each similarity and differences.
        """
        component = self.component
        
        # method 1 is relatively same with the other method but it does not contain difference feature and only contains similar features.
        # The similar features are calculated based on their standard deviation. If a feature is one standard deviation away from its overall 
        # mean for that feature, then the feature will be included in the similarities list as it
        if method == 1:
            for i in range(1,component+1):
                df_pca_kmeans = df_pca_kmeans.drop(['Component {}'.format(i)], axis = 1)
            
            ovr_df_pca_kmeans = df_pca_kmeans.drop('ClusterId',axis = 1)
            
            df_pca_kmeans_mean = ovr_df_pca_kmeans.mean()
            new_df = pd.DataFrame()
            new_df['overall mean'] = df_pca_kmeans_mean
            new_df['overall std'] = ovr_df_pca_kmeans.std()
            new_df['overall lower mean'] = new_df['overall mean'] - (new_df['overall std'])
            new_df['overall upper mean'] = new_df['overall mean'] + (new_df['overall std'])

            for i in range(1,50):
                df = df_pca_kmeans[df_pca_kmeans['ClusterId'] == i]
                df = df.drop('ClusterId',axis = 1)

                if i > num_of_cluster:
                    break
                else:
                    df_mean = df.mean()
                
                new_df['cluster {} mean'.format(i)] = df_mean

            for index, row in new_df.iterrows():
                for i in range(1, num_of_cluster+1):
                    if row['cluster {} mean'.format(i)] < row['overall lower mean']:
                        c1 = (row['overall lower mean'] - row['cluster {} mean'.format(i)]) / row['overall std']
                        row['cluster {} mean'.format(i)] = c1
                    elif row['cluster {} mean'.format(i)] > row['overall upper mean']:
                        c1 = (row['cluster {} mean'.format(i)] - row['overall upper mean']) / row['overall std']
                        row['cluster {} mean'.format(i)] = c1
                    else:
                        c1 = 0
                        row['cluster {} mean'.format(i)] = 0

            new_df = new_df.drop('overall lower mean', axis = 1)
            new_df = new_df.drop('overall upper mean', axis = 1)

            similarity_dictionary = {}
            difference_dictionary = {}
            similarity_output = {}
            difference_output = {}
            
            for i in range(1,num_of_cluster+1):
                temp = new_df.nlargest(10,'cluster {} mean'.format(i))
                temp = temp[temp['cluster {} mean'.format(i)] > 0]
                similarity_dictionary[i] = temp.index.tolist()
                difference_dictionary[i] = []
            
            for j in similarity_dictionary:
                df = df_pca_kmeans[df_pca_kmeans['ClusterId'] == j]
                temp_sim_output = []
                temp_dif_output = []

                for k in similarity_dictionary[j]:
                    mean = df[k].mean(axis=0)
                    variance = df[k].var(axis = 0)
                    temp_sim_output.append({'Feature' : k, 'Mean' : mean, 'Variance' : variance})
                
                similarity_output[j] = temp_sim_output
                difference_output[j] = temp_dif_output

            return similarity_output,difference_output

        else:
            for i in range(1,component+1):
                df_pca_kmeans = df_pca_kmeans.drop(['Component {}'.format(i)], axis = 1)
            
            output_similar = {}
            output_difference = {}

            # filter only the stores that fall within the identied clusterId
            for j in range(1,num_of_cluster+1):
                df = df_pca_kmeans[df_pca_kmeans['ClusterId'] == j]
                df = df.drop('ClusterId',axis = 1)
                # obtain the standard deviation for each feature in the dataframe
                df_std = df.std()

                similar = []
                difference = []
                
                # obtain the 5 most similar feature
                df_sim = df_std.sort_values(ascending = True).head(5)
                
                for i in df_sim.index:
                    similar.append(i)

                temp_sim_output = []

                # obtain the mean and variance for each selected feature 
                for k in similar:
                    mean = df[k].mean(axis = 0)
                    variance = df[k].var(axis = 0)
                    temp_sim_output.append({'Feature' : k, 'Mean' : mean, 'Variance' : variance})
                
                output_similar[j] = temp_sim_output

                # obtain the 5 most different feature
                df_dif = df_std.sort_values(ascending = False).head(5)
                
                for i in df_dif.index:
                    difference.append(i)

                temp_dif_output = []

                # obtain the mean and variance for each selected feature 
                for k in difference:
                    mean = df[k].mean(axis = 0)
                    variance = df[k].var(axis = 0)
                    temp_dif_output.append({'Feature' : k, 'Mean' : mean, 'Variance' : variance})
                
                output_similar[j] = temp_sim_output
                output_difference[j] = temp_dif_output

            return output_similar, output_difference

    def process(self, input):
        """
        input: file path to input.json file
        output: output.json data

        this function will take an input file, process the input data, perform clustering using k-means algorithm and output the sites with their clusterId. 
        Each clusterId will have their 5 similar features and 5 different features.
        """
        input_df = pd.read_json(input)
        df = input_df.copy()
        
        id_list = []
        
        for i in df['id']:
            id_list.append(i)

        df = input_df.iloc[:,1:]
        
        column_feature = []
        for i in df.columns:
            column_feature.append(i)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)

        # perform PCA and the scaled features to reduce dimension
        pca = PCA(n_components=self.component)
        pca.fit(scaled_features)
        # transform the data using pca
        scores_pca = pca.transform(scaled_features)
        kmeans_pca = KMeans(n_clusters=self.cluster,n_init=10,max_iter=300,random_state=42)
        kmeans_pca.fit(scores_pca)
        
        pca_kmeans = pd.concat([df.reset_index(drop = True),pd.DataFrame(scores_pca)], axis = 1)

        # how many components you used in PCA
        pca_component = self.component
        temp_component_list = []
        for i in range(1,pca_component+1):
            temp_component_list.append('Component {}'.format(i))
        pca_component = -pca_component
        pca_kmeans.columns.values[pca_component:] = temp_component_list
        
        temp = kmeans_pca.labels_
        for i in range(len(temp)):
            temp[i] += 1
        pca_kmeans['ClusterId'] = temp
        df_pca_kmeans = pca_kmeans
        df_pca_kmeans.insert(loc=0, column='id', value=id_list)
       
        # create a list for each clusters
        unique_cluster = df_pca_kmeans['ClusterId'].unique()
        output = {}
        for i in unique_cluster:
            output[i] = []

        for index,row in df_pca_kmeans.iterrows():
            output[int(row['ClusterId'])].append(str(row['id']))
        
        output_data = []

        output_data.append(
            {'Cluster' : 0,
            'Name' : 'Unclustered',
            'Ids' : [],
            'Similarities' : [],
            'Differences' : []}
            )

        # store each cluster details into the output_data
        # there are two methods in the differentiate, 1 means the latest method will be implemented and 
        # 0 or any other number means the old method will be implemented
        similar,different = self.differentiate(df_pca_kmeans, self.cluster,1)

        for i in similar:
            output_data.append(
            {'Cluster' : int(i),
            'Name' : str(i),
            'Ids' : output[int(i)],
            'Similarities' : similar[i],
            'Differences' : different[i]}
            )
      
        return output_data
