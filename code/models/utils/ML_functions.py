import logging
import operator
import pickle as pkl

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas
import pandas as pd
import seaborn
import seaborn as sns
import shap
### Library for oversampling Ref: https://imbalanced-learn.org/stable/over_sampling.html (N.B. per usare BorderlineSMOTE dobbiamo avere solo variabili numeriche)
from imblearn.over_sampling import SMOTENC
from matplotlib import pyplot
from scikitplot.metrics import plot_lift_curve
from scipy.stats import kruskal
from utils.google_utilities import create_logger

from .utils import read_pickle, save_pickle


class NewDataPreProcess:
    """
    A set of functions used for data pre-processing
    to prepare data for machine learning models.
    Example 
        dpp = NewDataPreProcess(data)
        x_all = dpp.prepare_data()
    """

    def __init__(self, df, 
                 etl_type='train', 
                 freq_treshold = 10, 
                 dummy_treshold=0.03, 
                 std_treshold = 0.05,
                 drop_autocorr = False,
                 corr_treshold = 0.9,
                 pkl_name = "preprocess.pickle",
                 target_col = None,
                 logger=None):

        if etl_type == 'train':
            self.pkl_filename =  pkl_name
            
            ### Drop zero variance columns
            n_cat = df.apply(pd.Series.nunique)
            df = df.loc[:, df.apply(pd.Series.nunique) > 1]
            col_zerovariance = []
            if len(n_cat[n_cat == 1])>0:
                col_zerovariance.append(n_cat[n_cat == 1])
                logger.info('Columns with zero variance (or all NULL values) dropped:')
                logger.info(n_cat[n_cat == 1].reset_index()['index'].to_list())
        else:
            self.pkl_filename =  pkl_name
            
        self.df = df
        self.etl_type = etl_type
        self.cat_vars = list(df.columns[pd.Series(df.columns).str.startswith('cat')])
        self.df_num = df.select_dtypes(include=['int8', 'int32', 'int64', 'float16','float64'])
        self.dummy_treshold = dummy_treshold
        self.freq_treshold = freq_treshold
        self.std_treshold = std_treshold
        self.corr_treshold = corr_treshold
        self.target_col=target_col
        self.drop_autocorr = drop_autocorr
        if logger:
            self.logger = logger
        else:
            self.logger = create_logger('ML_functions.py')

    def prepare_data(self, drop_first_dummy = False):
        ### Training data Preprocessing  
        if self.etl_type == 'train':
            
            df = self.df
            var_dict = {}
            col_other = {}
            col_categorical = []
            col_dummies = []
            
            ### Variables to be kept as Categorical (i.e. No One Hot Encoding) ###
            for col in self.cat_vars:
                freq = len(df[col].value_counts())
                if freq > self.freq_treshold:
                    df[col] = pd.Categorical(df[col])
                    col_categorical.append(col)
                else:
                    col_dummies.append(col)

            df_categorical = df[col_categorical]
            self.logger.info('Columns kept as categorical (i.e. no one hot encoding): {}'.format(col_categorical))

            ### Categories renamed 'Other' ###
            # for col in col_dummies:
            #     count = pd.value_counts(df.loc[:,col]) / df.shape[0]
            #     if len(count[count <= self.dummy_treshold]) > 0:
            #         mask = df.loc[:,col].isin(count[count > self.dummy_treshold].index)
            #         df.loc[~mask, col] = "Other"
            #         col_other[col]= count[count <= self.dummy_treshold].index
            #         self.logger.info("Values {} of variable {} renamed 'Other'".format(
            #             list(count[count <= self.dummy_treshold].index), col))
                    
            ### One Hot Encoding  ###
            # df_dummies = pd.get_dummies(df[col_dummies], drop_first=drop_first_dummy)        
            # self.logger.info('Columns converted in dummies: {}'.format(col_dummies))
# 
            ## Remove not significant dummies (low std) ###
            # dummies_dropped = df_dummies.std()[df_dummies.std() < self.std_treshold].index.values
            # df_dummies = df_dummies.drop(dummies_dropped, axis=1)
            # self.logger.info('Columns dummy dropped as not significant: {}'.format(dummies_dropped))
                    #    
            ## Final Dataset ###
            df_def = self.df_num.join(df_categorical)
            self.logger.info('Final list of variables: {}'.format(df_def.columns.to_list()))
            # 
            ### Drop varibales autocorrelated
            # if self.drop_autocorr == True:
            #     num_cols = df_def.select_dtypes(include=['int8', 'int32', 'int64', 'float','float64']).columns
            #     auto_corr_vars = self.drop_correlated(x_df=df_def, y_df=self.target_col, numeric_cols=num_cols, corr_coef=self.corr_treshold)
            #     df_def = df_def.drop(auto_corr_vars, axis=1)
            #     self.logger.info("Removed the following variables for autocorrelation: {}".format(auto_corr_vars))

            self.logger.info('Final list of variables: {}'.format(df_def.columns.to_list()))
            
            ### Save pickle with variables ###
            # var_dict['dummies'] = df_dummies.columns
            var_dict['categorical']=col_categorical
            var_dict['other']=col_other
            var_dict['final']=df_def.columns
            save_pickle(var_dict, self.pkl_filename)       

        ### Prediction data Preprocessing  
        else:
            df = self.df
            var_dict = read_pickle(self.pkl_filename)
            #col_dummies = var_dict['dummies']
            col_categorical= var_dict['categorical']
            col_other = var_dict['other']
            col_final =  var_dict['final']
            
            ### Variables to be kept as Categorical (i.e. No One Hot Encoding) ###
            for col in col_categorical:
                df[col] = pd.Categorical(df[col])
            df_categorical = df[col_categorical]

            ### Categories renamed 'Other' ###
            # dummies_vars = [item for item in self.cat_vars if item not in col_categorical]
            # for col in dummies_vars:
            #     if col in list(col_other.keys()):
            #         df[col] = np.where(df[col].isin(col_other[col]), 'Other', df[col])
            #         # print("Values {} of variable {} renamed 'Other'".format(col_other[col], col))

            ### One Hot Encoding  ###
            # df_dummies = pd.get_dummies(df[dummies_vars])     
            # df_dummies = df_dummies[col_dummies]      
            df_def = self.df_num.join(df_categorical)
            df_def = df_def[col_final]
                        
            self.logger.info('Final list of variables: {}'.format(df_def.columns.to_list()))

        self.logger.info('Preprocessing Completed!')
        return df_def   
    
    #### Drop the most auto-correlated numeric columns
    def drop_correlated(self, x_df, y_df, numeric_cols, corr_coef):
        """
        :param x_df: df without target variable
        :param y_df: target variable
        :param numeric_cols: numeric columns of x_df
        :param corr_coef: treshold for correlation
        :return: highly correlated features to remove
        """
        corr_coef = self.corr_treshold
        
        # Create a complete correlation matrix
        df_corr = x_df.loc[:, numeric_cols]
        df_corr['target__'] = y_df
        corr_matrix_full = df_corr.corr().abs()

        # Filter the correlation matrix to only contain variables correlated > corr_coeff
        corr_matrix = corr_matrix_full.copy()
        corr_matrix.values[[np.arange(corr_matrix.shape[0])]*2] = np.nan
        corr_matrix = corr_matrix.loc[corr_matrix.max(axis=1) > corr_coef, corr_matrix.max(axis=0) > corr_coef]

        # Drop perfectly correlated variables
        duplicate_cols = corr_matrix.loc[:, np.nanmax(np.triu(corr_matrix), axis=1)==1].columns
        corr_matrix.drop(columns=duplicate_cols, index=duplicate_cols, inplace=True)

        # Isolate correlation with target for all inter-correlated variables
        corr_vars = corr_matrix.columns
        target_corrs = corr_matrix_full.loc[corr_vars, 'target__']

        # Establish groups of correlated variables
        groups = []
        for col in corr_matrix.columns:
            group_temp = {}
            group_temp.update({col:target_corrs[col]})
            corr_cols = corr_matrix.loc[corr_matrix.loc[:,col]>corr_coef, :].index.to_list()
            for ccol in corr_cols:
                group_temp.update({ccol:target_corrs[ccol]})
            groups.append(group_temp)

        # Find best (most correlated with target) variable among each compound set of correlated variables
        while max([len(group) for group in groups]) > 1:

            # Find the best column per group as the "group_champion"
            group_champions = []
            for group in groups:
                group_champions.append(max(group.items(), key=operator.itemgetter(1))[0])
            group_champions = list(set(group_champions))

            # Establish groups of correlated champions 
            # (if any champions are correlated, the while loop will run until none are)
            groups = []
            df_corr_champs = corr_matrix.loc[group_champions, group_champions]
            for champion in group_champions:
                group_temp = {}
                group_temp.update({champion:target_corrs[champion]})

                corr_cols = df_corr_champs.loc[df_corr_champs.loc[:,champion]>corr_coef, :].index.to_list()

                for ccol in corr_cols:
                    group_temp.update({ccol:target_corrs[ccol]})

                groups.append(group_temp)

        # Define variables to drop based on compound correlation analysis
        final_vars = []
        for group in groups:
            final_vars.extend(list(group.keys()))
        vars_to_drop = list(corr_matrix.columns.difference(final_vars))
        vars_to_drop.extend(list(duplicate_cols))

        return vars_to_drop
    

###############
## Functions ##
###############
# Similarity analysis
def keep_topn_most_similar(sim_mat, index, top_n=10):
    """
    :param sim_mat: similarity matrix e.g. cosinus similarity
    :param index: column names used as index for pandas data frame
    :param top_n: the number of top most similar records to return for each entity
    :return: a pandas data frame that has top_n most similar records for each line
    """

    np.fill_diagonal(sim_mat, 0)
    df_sim = pd.DataFrame(sim_mat, index=index)
    a = np.array([df_sim[c].nlargest(top_n).index.values for c in df_sim])
    b = np.array([df_sim[c].nlargest(top_n).values for c in df_sim])
    a_df = pd.DataFrame(a, index=df_sim.index).reset_index()
    a_df = pd.melt(a_df, id_vars=['Qualitative ID'],
                   value_name='Qualitative IDs for most similar', var_name='top_n')
    b_df = pd.DataFrame(b, index=df_sim.index).reset_index()
    b_df = pd.melt(b_df, id_vars=['Qualitative ID'],
                   value_name='Cosinus similarity', var_name='top_n')
    return pd.merge(a_df, b_df, how='inner', on=['Qualitative ID', 'top_n'])

# Kruskal Wallis independence test
def apply_kw_test(df, ind_group1, ind_group2):
    ls_stats = []
    for col in df.columns:
        data1=df.loc[ind_group1, col]
        data2=df.loc[ind_group2, col]
        stat, p = kruskal(data1, data2, nan_policy='omit')
        ls_stats.append(stat)
    
    return pd.DataFrame(np.transpose([df.columns, ls_stats]), columns=['columns', 'stats'])

def plot_abs_SHAP_summary(
        x_valid,
        clf_all: lgbm.LGBMModel,       
        output_folder: str,
        logger=logging.getLogger('plot_abs_shap_values'),
        max_display=None):

    explainer = shap.TreeExplainer(clf_all)
    shap_values = explainer.shap_values(x_valid)

    if isinstance(shap_values, list):
        new_data_shap = shap_values[1]
    if isinstance(shap_values, np.ndarray):
        new_data_shap = shap_values

    df_shap = pandas.DataFrame(data=new_data_shap, columns=x_valid.columns, index=x_valid.index)
    logger.info("Plot abs Shap summary")
    # Get correlations and importances
    corrs = x_valid.corrwith(df_shap)
    shap_importances = df_shap.abs().mean()
    shap_importances.sort_values(ascending=False, inplace=True)
    corrs = corrs.loc[shap_importances.index]
    
    # Configure plot inputs
    if not max_display:
        max_display = len(shap_importances)
    shap_importances = shap_importances.iloc[:max_display]
    corrs = corrs.iloc[:max_display]
    norm = pyplot.Normalize(corrs.min(), corrs.max())
    colors = pyplot.cm.bwr(norm(corrs)) 
    scalar_color_map = pyplot.cm.ScalarMappable(cmap="bwr", norm=norm) #if you don't like my colors, change here
    scalar_color_map.set_array([])
    labels = shap_importances.index
    
    # Plot
    pyplot.figure(figsize=(20,len(shap_importances)*0.5))
    seaborn.barplot(x=shap_importances, y=numpy.arange(len(shap_importances)), orient='h', palette=colors)
    seaborn.despine(top=True, right=True)
    colorbar = pyplot.colorbar(scalar_color_map, aspect=50)
    colorbar.set_label('SHAP-feature correlation', fontsize=14)
    colorbar.set_ticks(numpy.arange(-1,1.25,0.25))
    pyplot.yticks(numpy.arange(len(shap_importances)), labels=labels, fontsize=12)
    pyplot.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=14)
    pyplot.title('SHAP feature importance', fontsize=18)
    pyplot.show()
    pyplot.savefig(output_folder + "/" + "importance_shap_values" + '.png')
    #plt.savefig(os.path.join(getcwd(),'shap_feature_imp.png'))
    
def lift_curve(y_pred_prob, y_valid, step = 0.1):
    print('####################### Lift score curve #######################')
    y_probas = np.zeros((len(y_pred_prob), 2))
    for i, p in enumerate(y_pred_prob):
        y_probas[i,] = [1-p,p]
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_lift_curve(y_valid,y_probas, ax=ax)
    ax.legend(loc='upper right')
    # plt.savefig(os.path.join(lgbm_path,'figures/LGBM_lift_curve.png'))

    print('Computing lift score values - Step={}'.format(step))
    aux_lift = pd.DataFrame()
    aux_lift['real'] = y_valid
    aux_lift['predicted'] = y_pred_prob
    aux_lift.sort_values('predicted',ascending=False,inplace=True)

    x_val = np.arange(step,1+step,step)
    #Calculate the ratio of ones in validation set
    ratio_ones = aux_lift['real'].sum() / len(aux_lift)
    y_val = []
    #Calculate for each x value its correspondent lift score value
    for x in x_val:
        num_data = int(np.ceil(x*len(aux_lift))) #ceil function returns the closest integer bigger than our number 
        data_here = aux_lift.iloc[:num_data,:]
        ratio_ones_here = data_here['real'].sum()/len(data_here)
        y_val.append(ratio_ones_here/ratio_ones)

    lift_frame = pd.DataFrame()
    lift_frame['percentile'] = x_val
    lift_frame['lift_score'] = y_val
    print(lift_frame)
    
def plot_density(df, feature_column, target_column, ordered=True, discrete=True):
    plt.figure(figsize=(12,5))
    df_tmp = df[(feature_column + ':'+ target_column).split(':')]
    if discrete:
        df_tmp[feature_column] = pd.Categorical(df_tmp[feature_column], ordered=True)
        sns.histplot(df_tmp, x=feature_column, hue=target_column, stat="density", discrete=discrete, common_norm=False)
    else: 
        sns.displot(df_tmp, x=feature_column, hue=target_column, kind="kde", fill=True, log_scale=True)

    del df_tmp
    plt.title('Density function of ' + feature_column, fontsize=16)
    plt.xlabel(None)
    plt.ylabel('Density', fontsize=16)
    plt.xticks(fontsize=14)
    plt.show()
    
#####################
## Diego Functions ##
#####################

### Function to compute uplift respect deciles
### The Classification report evaluates model precision using the probability threshold 0.5
### This approach is too restrictive in our case, because the campaign team will use the first three deciles to define the contact list
### That's why we need to evaluate the performance of the model by considering the first three deciles
### If we consider the first three deciles, the model have a recall of 75% and an uplift of 2.5 than a random model 

def model_uplift(df, tgt_var, score_var, logger):
    """
    Function to compute uplift respect deciles
    The Classification report evaluates model precision using the probability threshold 0.5
    This approach is too restrictive in our case, because the campaign team will use the first three deciles to define
    the contact list
    That's why we need to evaluate the performance of the model by considering the first three deciles
    If we consider the first three deciles, the model have a recall of 75% and an uplift of 2.5 than a random model
    :param df: dataframe (test or validation) containing flg target and score produced by the model
    :param tgt_var: flag identifying the target (0 vs 1)
    :param score_var: variables with score produced by the model
    :param logger:
    :return: dataframe containing deciles
    """
    if (tgt_var in df.columns)&(score_var in df.columns): 
        ### Decile variable
        df['decile_score'] = np.select(
            condlist = [df[score_var].isna(),
                        df[score_var]<=df[score_var].quantile(0.1), 
                        df[score_var]<=df[score_var].quantile(0.2),
                        df[score_var]<=df[score_var].quantile(0.3),
                        df[score_var]<=df[score_var].quantile(0.4),
                        df[score_var]<=df[score_var].quantile(0.5),
                        df[score_var]<=df[score_var].quantile(0.6),
                        df[score_var]<=df[score_var].quantile(0.7), 
                        df[score_var]<=df[score_var].quantile(0.8),
                        df[score_var]<=df[score_var].quantile(0.9),
                        df[score_var]>df[score_var].quantile(0.9)
                       ],
            choicelist = [99, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        )

        real_tgt_cum = 0
        uplift_cum = 0
        recall_cum = 0
        decile_df = pd.DataFrame()

        for idec in range(1, 11):
            df_decile = df[df['decile_score']==idec]
            real_tgt = df[df['decile_score']==idec][tgt_var].sum()
            real_tgt_cum += real_tgt

            ### Uplift ###
            # N.B. df[df[tgt_var]==1].shape[0]/df.shape[0] is the proportion of 1 on total dataset;
            # this can be considered as the precision of the most stupid model in which all data are marked as 1
            uplift = (real_tgt/df[df['decile_score']==idec].shape[0])/(df[df[tgt_var]==1].shape[0]/df.shape[0])
            uplift_cum = (real_tgt_cum/df[df['decile_score']<=idec].shape[0])/(df[df[tgt_var]==1].shape[0]/df.shape[0])

            ### Recall ###
            recall_perc = real_tgt/df[df[tgt_var]==1].shape[0]
            recall_perc_cum = real_tgt_cum/df[df[tgt_var]==1].shape[0]

            decile_row = {'Decile':idec, 'Tot Volumes':df_decile.shape[0], 'Real_Upg': real_tgt, 'Cumulative_Real_Upg': real_tgt_cum, 
                          'Uplift':uplift, 'Cumulative_Uplift':uplift_cum, 
                          'Recall':recall_perc, 'Cumulative_Recall':recall_perc_cum}
            decile_df = decile_df.append(decile_row, ignore_index = True)
        return(decile_df)
    else:
        logger.info('{} or {} variables not available in the dataset'.format(tgt_var, score_var))

        
def stacked_bar_charts(df, split_var, figsize=(5,3)):
    '''
    This function produce a stacked bar chart for each categorical variable/flag/dummy contained in df
    
    df: dataframe (test or validation) 
    split_var: variable to split the bars
    figsize: tuple to define the size of each plot
    '''
    categorical = list(df.select_dtypes(include=['object', 'category', 'bool', 'uint8']).columns)
    flags = list(df.columns[pd.Series(df.columns).str.startswith('flg')])
    all_vars = categorical + flags
    print(all_vars) 
    
    for ivar in all_vars:
        ### Stacked bar charts
        cross_tab_prop = pd.crosstab(index=df[split_var], columns=df[ivar], normalize="index")
        cross_tab_prop.plot(kind='bar', stacked=True, colormap='tab10', figsize=figsize)
        plt.title(ivar)
        plt.legend(loc="upper left", ncol=2)
        plt.xlabel("Type")
        plt.ylabel("Proportion")
        cross_tab = pd.crosstab(index=df[split_var],columns=df[ivar])
        
        for n, x in enumerate([*cross_tab.index.values]):
            for (proportion, y_loc) in zip(cross_tab_prop.loc[x],cross_tab_prop.loc[x].cumsum()):
                plt.text(x=n - 0.17,
                         y=(y_loc - proportion) + (proportion / 2),
                         s=f'{np.round(proportion * 100, 1)}%', 
                         color="black",
                         fontsize=12,
                         fontweight="bold")
        plt.show()


def correct_imbalances(x_tr,
                       y_tr, 
                       tgt_col = None,
                       undersampling = True, 
                       undersample = 300000, 
                       oversampling = True,
                       all_cat= True, 
                       export=False):

    '''
    This function use undersampling and oversampling techniques to deal with imbalanced training dataset

    x_tr: training x dataset
    y_tr: training y dataset
    tgt_col: name of target column 
    undersampling: if True, it randomly reduces the number of zeros    
    undersample: size of the new sample of zeros
    oversampling: if True, it increases the number of ones using SMOTENC algorithm
    all_cat: if True, flags and dummies variables are considered as categorical variables for the SMOTENC algorithm
    export: if True, it exports the new training dataset
    '''
    
    df_train  = pd.merge(x_tr, y_tr, how = 'left', right_index = True, left_index=True)
    
    ### Random undersampling of zeros 
    if undersampling == True:
        zeros = df_train[df_train[tgt_col]==0]
        ones = df_train[df_train[tgt_col]==1]
        new_zeros =zeros.sample(undersample)

        ### New x_tr e y_tr
        train_union = pd.concat([new_zeros, ones])
        x_tr = train_union.drop(tgt_col, axis=1)
        y_tr = train_union[tgt_col]
        print('Undersampling of Zeros Completed!')
       
    ### Oversampling of ones
    if oversampling == True:
        all_cat = True
        if all_cat:
            ### Categories + flags + dummies 
            categorical_features = list(x_tr.select_dtypes(include=['object', 'category', 'bool', 'uint8']).columns)
            flags = list(x_tr.columns[pd.Series(x_tr.columns).str.startswith('flg')])
            total_categorical = categorical_features + flags
        else:
            ### Only categorical variables  
            categorical_features = list(x_tr.select_dtypes(include=['object', 'category', 'bool']).columns)
            total_categorical = categorical_features 

        ### New x_tr e y_tr
        categorical_index = [x_tr.columns.get_loc(col) for col in total_categorical]
        smote_nc = SMOTENC(categorical_features=categorical_index, random_state=0)
        x_tr, y_tr = smote_nc.fit_resample(x_tr, y_tr)
        print('Oversampling of Ones Completed!')
        
    ### Export training dataset
    if export == True:
        export_training  = pd.merge(x_tr, y_tr, left_index=True, right_index=True)
        export_training.to_csv('/home/jupyter/ups_propensity_model/code/models/xtrain.csv', index = False)
    
    print('New X dataset has {} records'.format(x_tr.shape[0]))
    print('New Y dataset has {} records'.format(y_tr.shape[0]))
    return x_tr, y_tr


### New function for preprocessing ####
def preprocess (df, 
                etl_type = 'train', 
                freq_treshold = 5, 
                dummy_treshold = 0.03,
                std_treshold = 0.05,
                drop_first_dummy = False):
 
    '''
    input:
        df: dataset
        etl_type: train vs pred
        freq_treshold: default 10; if there are not more than 10 different values, convert text columns to categorical (i.e. no one hot encoding)
        dummy_treshold: 0.03 default; categories with lower percentage frequency than threshold will be renamed 'Other' before creating dummies

    '''
    ### Training data Preprocessing
    if etl_type == 'train':
        
        cat_file = open("categorical_var", "wb")       
        dummy_file = open("dummies_var.pickle", "wb")
        other_file = open("other_var.pickle", "wb")
        final_file = open("final_var.pickle", "wb")
        
        col_other = {}
        col_categorical = []
        col_dummies = []
        df_num = df.select_dtypes(include=['int8', 'int32', 'int64', 'float','float64'])
        
        ### Drop zero variance columns ###
        n_cat = df.apply(pd.Series.nunique)
        df = df.loc[:, df.apply(pd.Series.nunique) > 1]
        if len(n_cat[n_cat == 1])>0:
            col_zerovariance.append(n_cat[n_cat == 1])
            print('Columns with zero variance (or all NULL values) dropped:')
            print(n_cat[n_cat == 1])       
        
        ### Variables to be kept as Categorical (i.e. No One Hot Encoding) ###
        cat_vars = list(df.columns[pd.Series(df.columns).str.startswith('cat')])
        for col in cat_vars:
            freq = len(df[col].value_counts())
            if freq > freq_treshold:
                df[col] = pd.Categorical(df[col])
                col_categorical.append(col)
            else:
                col_dummies.append(col)
        
        df_categorical = df[col_categorical]
        print('Columns kept as categorical (i.e. no one hot encoding):', col_categorical)             
        
        ### Save pickle of categorical variables 
        pkl.dump(col_categorical, cat_file)               
        
        ### Categories renamed 'Other'
        for col in col_dummies:
            count = pd.value_counts(df.loc[:,col]) / df.shape[0]
            if len(count[count <= dummy_treshold]) > 0:
                mask = df.loc[:,col].isin(count[count > dummy_treshold].index)
                df.loc[~mask, col] = "Other"
                col_other[col]= count[count <= dummy_treshold].index
                print("Values {} of variable {} renamed 'Other'".format(list(count[count <= dummy_treshold].index), col))            
                
        ### Save pickle of variables with 'Other' aggregation
        pkl.dump(col_other, other_file)         
        
        ### One Hot Encoding  ###
        df_dummies = pd.get_dummies(df[col_dummies], drop_first=drop_first_dummy)        
        print('Columns converted in dummies:', col_dummies)
              
        ### Remove not significant dummies (low std)
        dummies_dropped = df_dummies.std()[df_dummies.std() < std_treshold].index.values
        df_dummies = df_dummies.drop(dummies_dropped, axis=1)
        print('Columns dummy dropped as not significant:', dummies_dropped)
        
        ### Save pickle of dummies
        pkl.dump(df_dummies.columns, dummy_file)  
        
        ### Final Dataset
        df_def = df_dummies.join(df_num).join(df_categorical)
        pkl.dump(df_def.columns, final_file)        
        print('Final list of variables:', df_def.columns)
        
    ### Prediction data Preprocessing  
    else:
        
        ### Carico i pickle del training
        cat_file = open("categorical_var", "rb")       
        dummy_file = open("dummies_var.pickle", "rb")
        other_file = open("other_var.pickle", "rb")     
        final_file = open("final_var.pickle", "rb") 
        
        col_dummies = pkl.load(dummy_file)
        col_categorical = pkl.load(cat_file)   
        col_other = pkl.load(other_file)   
        col_final = pkl.load(final_file)   

        cat_vars = list(df.columns[pd.Series(df.columns).str.startswith('cat')])
        df_num = df.select_dtypes(include=['int8', 'int32', 'int64', 'float','float64'])
        
        ### Variables to be kept as Categorical (i.e. No One Hot Encoding)
        for col in col_categorical:
            df[col] = pd.Categorical(df[col])
        df_categorical = df[col_categorical]
        
        ### Categories renamed 'Other'
        dummies_vars = [item for item in cat_vars if item not in col_categorical]
        for col in dummies_vars:
            if col in list(col_other.keys()):
                df[col] = np.where(df[col].isin(col_other[col]), 'Other', df[col])
                # print("Values {} of variable {} renamed 'Other'".format(col_other[col], col))
                      
        ### One Hot Encoding  
        df_dummies = pd.get_dummies(df[dummies_vars])     
        df_dummies = df_dummies[col_dummies]      
        df_def = df_dummies.join(df_num).join(df_categorical)
        df_def = df_def[col_final]
        print('Final list of variables:', df_def.columns)
        
    print('Preprocessing Completed!')
    return df_def   


def get_cv_scores_from_pipeline(pipeline, X, y, cv_method, scoring_metrics):

    scores = cross_validate(pipeline, X, y, cv = cv_method, scoring=scoring_metrics)
    scores_out = {k: v for k, v in scores.items() if k != "fit_time" and k != "score_time"}
    # Media delle metriche ottenuti sui k folds
    scores_avg_out = {k: v.mean().round(4) for k, v in scores_out.items()}
    print("AVG SCORES:\n", {k: v for k, v in scores_avg_out.items()})
    return scores_out, scores_avg_out