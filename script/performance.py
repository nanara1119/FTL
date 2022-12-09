import pandas as pd
import os

def output(result_list, path="", file_name = "", task = 0):
    """
    1. result_hospitalization_triage.csv / task1.csv

    """
    result_df = pd.DataFrame(result_list, columns=['Model', 'auroc', 'ap', 'sensitivity', 'specificity', 'threshold', 
                                                'lower_auroc', 'upper_auroc', 'std_auroc', 'lower_ap', 'upper_ap', 
                                                'std_ap', 'lower_sensitivity', 'upper_sensitivity', 'std_sensitivity',
                                                'lower_specificity', 'upper_specificity', 'std_specificity', 'runtime'])
    result_df.to_csv(os.path.join(path, f'result_{file_name}.csv'), index=False)

    result_df = result_df.round(3)
    formatted_result_df = pd.DataFrame()
    formatted_result_df[['Model', 'Threshold']] = result_df[['Model', 'threshold']]
    formatted_result_df['AUROC'] = result_df['auroc'].astype(str) + ' (' + result_df['lower_auroc'].astype(str) + \
                                '-' + result_df['upper_auroc'].astype(str) + ')'
    formatted_result_df['AUPRC'] = result_df['ap'].astype(str) + ' (' + result_df['lower_ap'].astype(str) + \
                                '-' + result_df['upper_ap'].astype(str) + ')'
    formatted_result_df['Sensitivity'] = result_df['sensitivity'].astype(str) + ' (' + result_df['lower_sensitivity'].astype(str) + \
                                        '-' + result_df['upper_sensitivity'].astype(str) + ')'
    formatted_result_df['Specificity'] = result_df['specificity'].astype(str) + ' (' + result_df['lower_specificity'].astype(str) + \
                                        '-' + result_df['upper_specificity'].astype(str) + ')'
    formatted_result_df[['Runtime']] = result_df[['runtime']]
    formatted_result_df.to_csv(os.path.join(path, f'task_{file_name}.csv'), index=False)
    formatted_result_df
