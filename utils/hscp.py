import numpy as np
import pandas as pd
import scipy
from sklearn.decomposition import PCA
import warnings
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress all warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=tf.compat.v1.DeprecationWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class HSPC:
    def __init__(self,excel_file_path):
        df = pd.read_excel(excel_file_path,sheet_name=['Dataset 1', 'Dataset 2'],index_col=None)
        dataset1_df = df["Dataset 1"]
        dataset2_df = df["Dataset 2"]
        dataset_df_meta = pd.concat([dataset1_df.iloc[:,:15],dataset2_df.iloc[:,:15]],axis=0, ignore_index=True)
        labels1, labels2 = self._find_closest_pairs(dataset1_df,dataset2_df)

        result_df = self._combine_dataframes(dataset1_df, dataset2_df, zip(labels1, labels2))
        result_df.reset_index(drop=True, inplace=True)
        dataset_df_meta.reset_index(drop=True, inplace=True)
        self.dataset_df = pd.concat([dataset_df_meta,result_df],axis=1)
        print(f'data loaded and combined consisting of {self.dataset_df.shape[0]} observations and {self.dataset_df.shape[1]} features')

    def select_wavelegnth_range(self,wavelength_range):
        vars = np.array(self.dataset_df.keys())
        wave_lengths_ids = np.where(np.vectorize(lambda x: not isinstance(x, str))(vars))[0]
        # print(wave_lengths_ids)
        wave_lengths = vars[wave_lengths_ids]
        working_wave_length_ids = np.where(np.logical_and(wave_lengths > wavelength_range[0],wave_lengths < wavelength_range[1]))[0]
        self.working_wave_length = wave_lengths[working_wave_length_ids]
        self.df_working_wave_length_ids = wave_lengths_ids[working_wave_length_ids]
        print(f'number of wavelegnths falling into range:{len(self.working_wave_length)}')

    def batch_outlier_removal_persample(self):
        unique_sample_ids = np.unique( self.dataset_df['Sample ID'])
        for s in unique_sample_ids:
            current_samp_index = np.where( self.dataset_df['Sample ID']== s)[0]
            current_back_sample_index = current_samp_index[np.where(( self.dataset_df['Background'][current_samp_index]==1))[0]]
            current_samp_index = np.delete(current_samp_index,current_samp_index==current_back_sample_index)
            # scan_ids =  np.array(dataset_df.iloc[current_samp_index]['ScanIndex'])

            SX = self.dataset_df.iloc[current_samp_index][ self.working_wave_length]
            inliers_idx,outlier_idx =  self._zscore_ourlierRemoval(SX,threshold=2)
            global_outlier_idx=current_samp_index[outlier_idx]
            self.dataset_df = self.dataset_df.drop(global_outlier_idx)
            self.dataset_df.reset_index(drop=True, inplace=True)

    def write_combined_dataset(self,file_path='dataset_df.csv'):
        self.dataset_df.to_csv(file_path)

    def compute_and_preprocess_absorbance(self,smooth_data=True,normalize_data=True,overall_outlier_removal=True,apply_pca=True):
        dataset1_absorbance_df = self.dataset_df.iloc[:,:15]
        dataset1_absorbance_df = dataset1_absorbance_df.drop('Background',axis=1)
        unique_sample_ids = np.unique(self.dataset_df['Sample ID'])
        self.absorbance_df = pd.DataFrame()
        for s in unique_sample_ids:
            current_samp_index = np.where(self.dataset_df['Sample ID']== s)[0]
            current_back_sample_index = current_samp_index[np.where((self.dataset_df['Background'][current_samp_index]==1))[0]]
            current_samp_index = np.delete(current_samp_index,current_samp_index==current_back_sample_index)
            scan_ids =  np.array(self.dataset_df.iloc[current_samp_index]['ScanIndex'])

            SX = self.dataset_df.iloc[current_samp_index][self.working_wave_length]
                
            BX = self.dataset_df.iloc[current_back_sample_index][self.working_wave_length]
            batch_absorbance = self._absorbance_estimator(SX.values,BX.values)
            if smooth_data:
                batch_absorbance = self._data_smoother(batch_absorbance,window_size=5)
            if normalize_data:
                batch_absorbance = self._data_normalizer(batch_absorbance)
            # cols=np.concat()
            batch_absorbance_df = pd.DataFrame({'Sample ID':[s]*len(current_samp_index),'ScanIndex':scan_ids})
            batch_absorbance_df = pd.concat([batch_absorbance_df,pd.DataFrame(batch_absorbance,columns=self.working_wave_length)],axis=1)
            self.absorbance_df = pd.concat([self.absorbance_df, batch_absorbance_df]) 

        self.absorbance_after_batch = self.absorbance_df
        ## remove the backgrounds from features
        self.dataset_df_new = self.dataset_df.drop(np.where(self.dataset_df['Background'] == 1.0)[0])
        self.dataset_df_new = self.dataset_df_new.drop('Background',axis=1)
        self.absorbance_df_before_outlier_removal = self.absorbance_df
        if overall_outlier_removal:
            inliers_idx,outlier_idx = self._zscore_ourlierRemoval(self.absorbance_df.iloc[:,2:],threshold=1.5)
            print('outlier_samples are\n',self.absorbance_df.iloc[outlier_idx,0:2])
            self.absorbance_df = self.absorbance_df.iloc[inliers_idx,:]
            self.absorbance_df_after_outlier_removal = self.absorbance_df
            self.dataset_df_new = self.dataset_df_new.iloc[inliers_idx,:]
            if apply_pca:
                self.before_pca = self.absorbance_df.iloc[:,2:]
                absorbance_df_lowdim,self.pca_comps = self._PCA_estimator(self.absorbance_df.iloc[:,2:],num_components=64)
                self.output_of_pca = absorbance_df_lowdim
            else:
                absorbance_df_lowdim = self.absorbance_df.iloc[:,2:]
            self.absorbance_df_org=self.absorbance_df
            # tobeupdated_columns=absorbance_df.columns[2:]
            self.absorbance_df.reset_index(drop=True, inplace=True)
            absorbance_df_lowdim.reset_index(drop=True, inplace=True)
            self.absorbance_df = pd.concat([self.absorbance_df.iloc[:,0:2],absorbance_df_lowdim],axis=1)
        print(f'absorbance dataset create consisting of {self.absorbance_df.shape[0]} observations and {self.absorbance_df.shape[1]} features')
        return self.absorbance_df
    
    def build_training_dataset(self):
        meta_data_df = self.dataset_df_new.iloc[:,:14]
        tmp=meta_data_df['ScanIndex']
        meta_data_df = meta_data_df.drop('ScanIndex',axis=1)
        meta_data_df.insert(1,'ScanIndex',tmp)

        self.target_df=meta_data_df['Total Organic Carbon [TOC]']
        meta_data_df = meta_data_df.drop('Total Organic Carbon [TOC]',axis=1)
        meta_data_df.reset_index(drop=True, inplace=True)
        absorbance_df_sliced = self.absorbance_df.iloc[:,2:]
        absorbance_df_sliced.reset_index(drop=True, inplace=True)
        features_df = pd.concat([meta_data_df,absorbance_df_sliced],axis=1)
        features_df.to_csv('features_df.csv', index=False)
        self.absorbance_features = features_df.values[:,13:]
        self.including_metadata_features = features_df.values[:,2:]

        print('training dataset with only absorbance data shape:',self.absorbance_features.shape)
        print('training dataset including absorbance data shape:',self.including_metadata_features.shape)

    def train_model(self,features,target,learning_rate = 0.001,epochs=30):

        def regress_NN_model(num_features,dropout=0.2,ini = 'glorot_uniform'):
            I = layers.Input(shape=(num_features,))

            A1 = layers.Dense(32,kernel_regularizer=regularizers.l2(0.01),kernel_initializer=ini)(I)
            A1 = layers.BatchNormalization()(A1)
            A1 = layers.Activation('relu')(A1)
            A1 = layers.Dropout(dropout)(A1)

            A2 = layers.Dense(32,kernel_regularizer=regularizers.l2(0.01),kernel_initializer=ini)(A1)
            A2 = layers.BatchNormalization()(A2)
            A2 = layers.Activation('relu')(A2)
            A2 = layers.Dropout(dropout)(A2)

            A3 = layers.Dense(16,kernel_regularizer=regularizers.l2(0.01),kernel_initializer=ini)(A2)
            A3 = layers.BatchNormalization()(A3)
            A3 = layers.Activation('relu')(A3)
            # A3 = layers.Dropout(dropout)(A3)

            # A4 = layers.Dense(16,kernel_regularizer=regularizers.l2(0.01),kernel_initializer=ini)(A3)
            # A4 = layers.BatchNormalization()(A4)
            # A4 = layers.Activation('relu')(A4)

            # A5 = layers.Dropout(dropout)(A5)

            O = layers.Dense(1, activation='sigmoid')(A3)
            model = models.Model(inputs=I, outputs=O)
            return model

        features = self.absorbance_features
        self.target = np.array(self.target_df)
        features_tf= tf.constant(features, dtype=tf.float32)
        target_tf = tf.constant(target,dtype=tf.float32)
        num_features = features.shape[1]

        self.regress_model = regress_NN_model(num_features)
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.regress_model.compile(optimizer=adam_optimizer, loss='mae', metrics=['mae'])
        self.history = self.regress_model.fit(features_tf, target_tf, epochs=epochs, batch_size=16, validation_split=0.1)
        return self.history

    def plot_training_perfomance(self,history):
        plt.figure(figsize=(12, 6))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation mean absolute error (mae) values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Mean Absolute Error (MAE)')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()

    def predict(self,test_observation):
        test_observation_tf = tf.constant(test_observation, dtype=tf.float32)
        test_data=tf.expand_dims(test_observation_tf,axis=0)
        return self.regress_model.predict(test_data)

    def _find_closest_pairs(self,df1, df2):
    # Get column labels from both DataFrames
        labels1 = np.array(df1.columns)[15:]
        labels2 = np.array(df2.columns)[15:]
        # Calculate pairwise Euclidean distances between column labels
        distances = np.abs(labels1[:, np.newaxis] - labels2)
        # Find the indices of the closest pairs
        closest_indices = np.argmin(distances, axis=1)
        return labels1, labels2[closest_indices]

    def _combine_dataframes(self,df1, df2, closest_labels):
        # Create a new DataFrame with averaged values for closest pairs
        combined_df = pd.DataFrame()

        for label1, label2 in closest_labels:
            new_label = round(np.mean([label1,label2]))
            df1[label1] 
            df2[label2]
            combined_df[new_label] = pd.concat([df1[label1],df2[label2]],axis=0)

        return combined_df
    
    def _absorbance_estimator(self,SX,BX):
        A = SX/BX
        absorbance = np.log(1/A)
        return absorbance

    def _data_smoother(self,A,window_size=5):
        smoothed_A = np.apply_along_axis(lambda x: scipy.signal.convolve(x, np.ones(window_size)/window_size, mode='same'), axis=1, arr=A)
        # print(f'features of samples are smoothed with a window size of {window_size}')
        return smoothed_A

    def _data_normalizer(self,A):
        norms_a = np.linalg.norm(A, axis=1, ord=2, keepdims=True)
        # print("features are normalized by norm2")
        return A/norms_a

    def _zscore_ourlierRemoval(self,df, threshold=2):
        z_scores = np.abs((df - df.mean()) / df.std())
        outlier_idx = np.where((z_scores >= threshold).any(axis=1))[0]
        inliers_idx = np.where((z_scores < threshold).all(axis=1))[0]
        # print(f'outlier removal  with threshold {threshold}')
        return inliers_idx,outlier_idx

    def _PCA_estimator(self,df, num_components):
        # Standardize the data (optional but recommended for PCA)
        standardized_data = (df - df.mean()) / df.std()

        # Create a PCA instance and fit_transform the data
        pca = PCA(n_components=num_components)
        principal_components = pca.fit_transform(standardized_data)

        # Create a DataFrame with the principal components
        pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i}' for i in range(1, num_components + 1)])
        print(f'dimension of features are reduced to {num_components} componnents')

        return pc_df, pca
    
    def vizualize_pca_perfromance(self):

        plt.figure(figsize=(22, 8))
        sns.boxplot(data=self.before_pca.iloc[:,2:], palette="Set3")
        plt.title("distribution of absorbance before PCA")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.ylim(0, 0.150000) 
        plt.show()

        plt.figure(figsize=(22, 8))
        sns.boxplot(data=self.before_pca.iloc[:,2:], palette="Set3")
        plt.title("distribution of absorbance feature selection After PCA")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.show()

    def vizualize_outlier_removal_perfromance(self):

        plt.figure(figsize=(22, 8))
        sns.boxplot(data=self.absorbance_df_before_outlier_removal.iloc[:,2:], palette="Set3")
        plt.title("distribution of absorbance before outlier removal")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.show()

        plt.figure(figsize=(22, 8))
        sns.boxplot(data=self.absorbance_df_after_outlier_removal.iloc[:,2:], palette="Set3")
        plt.title("distribution of absorbance after outlier removal")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.show()

    def vizualize_smoother_performance(self):
        self.compute_and_preprocess_absorbance(smooth_data=False,normalize_data=True,overall_outlier_removal=True,apply_pca=True)
        absorbance_without_smoother =  self.absorbance_after_batch.iloc[:,2:]
        self.compute_and_preprocess_absorbance(smooth_data=True,normalize_data=True,overall_outlier_removal=True,apply_pca=True)
        absorbance_with_smoother =  self.absorbance_after_batch.iloc[:,2:]
        absorbance_without_smoother_arr = np.array(absorbance_without_smoother)
        absorbance_with_smoother_arr =  np.array(absorbance_with_smoother)
        absorbance_without_smoother_arr_var = np.var(absorbance_without_smoother_arr,axis=0)
        absorbance_with_smoother_arr_var =  np.var(absorbance_with_smoother_arr,axis=0)

        max_y=np.max(absorbance_without_smoother_arr_var)
        plt.figure(figsize=(22, 8))
        plt.bar(range(len(absorbance_without_smoother_arr_var)), absorbance_without_smoother_arr_var)
        plt.title('variance of absorbance accross all features without smoother')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.show()

        plt.figure(figsize=(22, 8))
        plt.bar(range(len(absorbance_with_smoother_arr_var)), absorbance_with_smoother_arr_var)
        plt.title('variance of absorbance accross all features with smoother')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.ylim(0, max_y) 
        plt.show()