import argparse
import pandas as pd

diseases = ('aneurysm-of-heart', 'angina-pectoris', 'aortic-valve-disorders', 'arrhythmia_tachycardia',
            'atrial-fibrillation_flutter', 'atrioventricular-block', 'cardiac-arrest', 'cardiomegaly',
            'cardiomyopathy', 'coronary-artery-aneurysm', 'coronary-heart-disease', 'embolism-and-thrombosis-of-vena-cava',
            'fascicular-block', 'heart_cardiac-problem', 'heart-failure', 'high-cholesterol',
            'hypertension', 'hypotension', 'intracardiac-thrombosis', 'left-ventricular-failure',
            'mitral-problems', 'myocardial-infarction', 'myocarditis', 'other-heart-diseases',
            'pericardial-problem', 'premature-depolarisation', 'Raynauds-syndrome', 'Tawara-branches',
            'thoracic-aortic-aneurysm', 'type-2-diabetes')


def process_arguments():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dir', action="store", default="", help="Fully-qualified path of the directory containining pkl results of ML analyses (it will also be used to save the csv files).")
    parser.add_argument('--files', action="store", default="", help="Semicolon-separated list of input pkl file names (with or without extension).")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

def get_tsv_id(file_name):
    # Return tsv_id field (to be saved in csv) from file name
    tsv_id_split = file_name.split('_')
    tsv_id = tsv_id_split[-3] + '_' + tsv_id_split[-2]
    for disease in diseases:  # diseases do not respect the "rule" above for naming
        if disease in file_name:
            tsv_id = disease
    return tsv_id

def is_add_feat(keys):
    # Returns True if the pkl results also consider additional features, otherwise False
    for key in keys:  # iterate over dictionary keys
        if '_AddFeat_' in key:
            return True  # True if at least one key is about additional features
    return False  # False if no key is about additional features


def main():
    args, unknown_args = process_arguments()

    dir = args.dir if args.dir.endswith('/') else args.dir + '/'  # get directory path (ending with /)
    file_combos = args.files.split(';')  # get list of pkl files to convert
    for file in file_combos:  # iteratively process pkl files
        file = file[0: -4] if file.endswith('.pkl') else file  # current file name, without extension
        in_path = dir + file + '.pkl'  # fully-qualified path of the current pkl file
        dic = pd.read_pickle(in_path)  # read pkl file

        col = ['tsv_id', 'Trait', 'Model', 'Norm', 'customL1', 'L1_penalization', 'CV_OR_HeldoutTest', 'Fold_num', 
               'pkl', 'Intercept', 'Coefficients', 'SignificantCoefficients', 
               'NumCoefficients', 'NumSignificantCoefficients', 'Classification_OR_Regression', 
               'MSE', 'R-squared', 
               'ClassifRprt_accuracy', 'ClassifRprt_precision', 'ClassifRprt_recall', 'ClassifRprt_f1-score', 'AUC',
               'TrainSize', 'TestSize']  # information to save in csv
        df = pd.DataFrame(columns=col)  # blank dataframe with information to fill

        contains_add_feat = is_add_feat(dic.keys())  # boolean for results also containing additional features (or not)

        # Navigate the structure of the pkl file and extract relevant information
        for primary_key in dic.keys():  # iterate over model results
            # primary_key refers to the model
            if ('_FeatureSelection' not in primary_key) and ('_RFE' not in primary_key) and ('_Assoc_' not in primary_key):  # FS, RFE and Association results not saved in the csv file
                if (not contains_add_feat) or (contains_add_feat and '_AddFeat_' in primary_key):  # if pkl file is also about additional features, save only the results with those
                    primary_key_split = primary_key.split('_')
                    model = (primary_key_split[-3] if '_customL1_' in primary_key else primary_key_split[-2]) if (not contains_add_feat) else (primary_key_split[-4] if '_customL1_' in primary_key else primary_key_split[-3])  # type of model
                    norm = '_norm_' in primary_key  # True for normalized phenotype, False otherwise
                    trait = primary_key[:primary_key.index('_norm_') if norm else primary_key.index('_' + model + '_')]
                    regr = not('CAT' in trait)  # True if regression, False if classification
                    l1_pen = 'OptimalAlpha' if regr else 'OptimalC'  # name for L1 penalty field
                    for secondary_key in dic[primary_key].keys():
                        # secondary_key refers to the fold
                        for tertiary_key in dic[primary_key][secondary_key].keys():
                            # tertiary_key refers to the results
                            if tertiary_key.startswith('SignificantCoefficients'):
                                sign_coef = tertiary_key  # sign_coef is the field referring to the significant coefficients
                        df_curr = pd.DataFrame({'tsv_id': get_tsv_id(file),
                                                'Trait': trait,
                                                'Model': model,
                                                'Norm': norm,
                                                'CV_OR_HeldoutTest': 'HeldoutTest' if secondary_key=='heldoutTest' else 'CV',
                                                'Fold_num': None if secondary_key=='heldoutTest' else secondary_key.split('_')[-1],
                                                'customL1': '_customL1_' in primary_key,
                                                'L1_penalization': dic[primary_key][secondary_key][l1_pen] if '_Lasso' in primary_key else None,
                                                'pkl': in_path,
                                                'Intercept': '[' + primary_key + '][' + secondary_key + '][Intercept]',
                                                'Coefficients': '[' + primary_key + '][' + secondary_key + '][Coefficients]',
                                                'SignificantCoefficients': '[' + primary_key + '][' + secondary_key + '][' + sign_coef + ']',
                                                'NumCoefficients': len(dic[primary_key][secondary_key]['Coefficients']),
                                                'NumSignificantCoefficients': len(dic[primary_key][secondary_key][sign_coef]),
                                                'Classification_OR_Regression': 'Regression' if regr else 'Classification',
                                                'MSE': dic[primary_key][secondary_key]['MSE_TestSet'] if regr else None,
                                                'R-squared': dic[primary_key][secondary_key]['R-squared_TestSet'] if regr else None,
                                                'ClassifRprt_accuracy': None if regr else dic[primary_key][secondary_key]['ClassifRprt_TestSet'].loc['accuracy', 'support'],
                                                'ClassifRprt_precision': None if regr else dic[primary_key][secondary_key]['ClassifRprt_TestSet'].loc['weighted avg', 'precision'],
                                                'ClassifRprt_recall': None if regr else dic[primary_key][secondary_key]['ClassifRprt_TestSet'].loc['weighted avg', 'recall'],
                                                'ClassifRprt_f1-score': None if regr else dic[primary_key][secondary_key]['ClassifRprt_TestSet'].loc['weighted avg', 'f1-score'],
                                                'AUC': None if regr else dic[primary_key][secondary_key]['AUC'],
                                                'TrainSize': dic[primary_key][secondary_key]['TrainSize'],
                                                'TestSize': dic[primary_key][secondary_key]['TestSize']},
                                            index=[0])
                        df = pd.concat([df, df_curr], ignore_index=True)

        df.to_csv(dir + file + '.csv', index=False, index_label=False)


if __name__ == "__main__":
    main()