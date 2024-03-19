import logging
import os
from lightgbm import LGBMModel

import lightgbm as lgbm
import shap
import pandas
import numpy
import matplotlib
from matplotlib import pyplot
import seaborn
import sklearn
seaborn.set()


def plot_histograms_with_target(
        input_dataframe: pandas.DataFrame, 
        columns: list, 
        target_column: str, 
        output_folder: str,
        suffix: str="",
        logger=logging.getLogger('plot_histograms')
    ) -> None:
    """
    Save in output_folder one histogram plot per input_dataframe column in the list columns.
    Each hist_plot has the hue configured by the target_column
    """
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    pyplot.rc('font', size=8)          # controls default text sizes
    pyplot.rc('axes', titlesize=8) 
    pyplot.rc('axes', labelsize=8)    # fontsize of the x and y labels
    pyplot.rc('xtick', labelsize=8)    # fontsize of the tick labels
    pyplot.rc('ytick', labelsize=8)    # fontsize of the tick labels
    pyplot.rc('legend', fontsize=8)    # legend fontsize
    for column_name in columns:
        pyplot.figure(figsize=(5,3))
        locs, labels = pyplot.xticks()
        pyplot.setp(labels, rotation=60)
        pyplot.title(column_name)
        seaborn.histplot(input_dataframe, x=column_name, stat="density", hue=target_column, common_norm=False)
        pyplot.tight_layout()
        pyplot.savefig(output_folder+"/histplot_{column_name}{suffix}.png".format(column_name=column_name, suffix=suffix), dpi=300)
    logger.info("Successfully created histplots")


def plot_shap_values(
        input_dataframe: pandas.DataFrame,
        input_model: lgbm.LGBMModel,
        n_shaps: int,
        type_shap: str,
        importance_treshold: float,
        output_folder: str,
        output_filename_prefix: str,
        logger
    ) -> None:
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    data_shap = input_dataframe[:n_shaps]
    logger.info(f"Data for Shap shape: {data_shap.shape}")
    
    shap_values = shap.TreeExplainer(input_model).shap_values(data_shap)

    vals = numpy.abs(shap_values).mean(0)
    df_shap = pandas.DataFrame(data=shap_values[1], columns=data_shap.columns, index=data_shap.index)
    ### Plot Shap Values
    pyplot.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[1], df_shap, plot_type=type_shap)  
    pyplot.savefig(output_folder + "/" + output_filename_prefix + type_shap + '.png', dpi=200, bbox_inches='tight')
    ### Log low importance features according to shap 
    feature_importance = pandas.DataFrame(list(zip(input_dataframe.columns, sum(vals))), columns=['Feature','Feature_importance'])
    feature_importance.sort_values(by=['Feature_importance'], ascending=False, inplace=True)
    feature_importance['perc_importance'] = feature_importance['Feature_importance'].apply(lambda x: x/feature_importance['Feature_importance'].sum())
    feature_to_remove = feature_importance[feature_importance['perc_importance'] < importance_treshold]['Feature']
    logger.info('Low importance features (according to shap) that must be removed from the model:\n{}'.format(feature_to_remove))
    ### Dependency diagrams
    logger.info('Building SHAP dependence plots for all features')
    for i, feature in enumerate(feature_importance['Feature']):
        shap.dependence_plot(
            feature,
            shap_values[1],
            data_shap,
            interaction_index=None
        )
        pyplot.savefig(output_folder + "/" + "{i}_{name}_".format(i=i, name=feature) + output_filename_prefix + '.png', dpi=200, bbox_inches='tight')