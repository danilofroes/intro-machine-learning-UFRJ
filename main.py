import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from skopt import BayesSearchCV

class ClassificationEvaluator:
    def __init__(self, output_path):
        self.output_path = output_path

    def _plot_confusion_matrix(self, key, y_true, y_pred, cross_val=False, all_conf_matrices=[]):

        if cross_val and y_pred == None and y_true == None:
            cm = np.mean(all_conf_matrices, axis=0)
        else:
            cm = confusion_matrix(y_true, y_pred)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Inadimplente', 'Inadimplente'])
        disp.plot(cmap=plt.cm.Blues)

        os.makedirs(self.output_path, exist_ok=True)
        plt.savefig(os.path.join(self.output_path, f'confusion_matrix_{key}.jpg'))
        plt.close()

    def evaluate(self, key, fold_results):
        scores = []  # Lista para armazenar os scores de cada fold

        for fold in fold_results:
            y_true = fold['y_true']
            y_pred = fold['y_pred']

            score = {'accuracy' : accuracy_score(y_true, y_pred)}
            scores.append(score)

        if len(fold_results) > 1:
            all_conf_matrices = [] 
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            all_conf_matrices.append(conf_matrix)

            self._plot_confusion_matrix(key, None, None, cross_val=True, all_conf_matrices=all_conf_matrices)
        else:
            self._plot_confusion_matrix(key, y_true, y_pred)

        return scores

class ModeloML:
    def __init__(self):
        self.pipeline_registry = {}

    def load_data(self, df_treinamento, df_predicao):

        y_col = 'inadimplente'  # Coluna target
        x_cols = df_treinamento.drop(columns=[y_col]).columns  # Colunas de features

        # Divide os dados de treino em X (features) e y (target)
        self.y_train = df_treinamento[y_col]
        self.X_train = df_treinamento[x_cols]

        X_predict = df_predicao.copy() # Cópia para evitar alterações no DataFrame original

    def holdout(self, test_size=0.2, random_state=42):

        holdout_results = [] # Lista para armazenar os resultados da divisão holdout

        # Divide os dados de treino em X e y
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=test_size, random_state=random_state)

        # Armazena os resultados da divisão em um dicionário
        holdout_results.append({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })

        key = 'holdout' # Chave para armazenar os resultados

        # Armazena os resultados no registro de pipeline para poder ser acessado posteriormente
        self.pipeline_registry[key] = {
            'split_data' : holdout_results
        }

        return holdout_results
    
    def KFold_cross_validation(self, n_splits=5, shuffle=True, random_state=42):

        folds_results = [] # Lista para armazenar os folds criados

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) # Cria um objeto KFold para dividir os dados em k-folds

        # Divide os dados de treino em k-folds
        for train_index, test_index in kf.split(self.X_train, self.y_train):
            X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
            y_train, y_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
            folds_results.append({
                'fold': len(folds_results) + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })

        key = 'KFold' 

        self.pipeline_registry[key] = {
            'split_data': folds_results
        }

        return folds_results
    
    def KFold_cross_validation_stratified(self, n_splits=5, shuffle=True, random_state=42):

        folds_results = [] # Lista para armazenar os folds criados

        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state) # Cria um objeto StratifiedKFold para dividir os dados em k-folds estratificados

        for train_index, test_index in skf.split(self.X_train, self.y_train):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]
            folds_results.append({
                'fold': len(folds_results) + 1,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })

        key = 'KFold_stratified'

        self.pipeline_registry[key] = {
            'split_data': folds_results
        }

        return folds_results

    def scale_data(self, num_cols, scalers = []):

        new_pipeline_registry = {}

        for key in self.pipeline_registry.keys():
            for scaler in scalers:
                num_scaler = scaler() # Cria uma instância do scaler

                key_scaler = f'scaled_{type(scaler).__name__}_{key}'
                new_pipeline_registry[key_scaler] = {
                    **self.pipeline_registry[key],
                    'scaler': num_scaler
                }

                for fold_data in self.pipeline_registry[key]['split_data']:
                    # for fold_data in data.values():
                    X_num_train = fold_data['X_train'][num_cols]
                    X_num_test = fold_data['X_test'][num_cols]

                    fold_data['X_train'][num_cols] = num_scaler.fit_transform(X_num_train)
                    fold_data['X_test'][num_cols] = num_scaler.transform(X_num_test)

        self.pipeline_registry = new_pipeline_registry # Atualiza o registro de pipeline com os dados escalados

    def encode_data(self, cat_cols, encoders = []):
            
        new_pipeline_registry = {}

        for key in self.pipeline_registry.keys():
            for encoder in encoders:
                cat_encoder = encoder() # Cria uma instância do encoder
                
                key_encoder = f'encoded_{type(encoder).__name__}_{key}'
                new_pipeline_registry[key_encoder] = {
                    **self.pipeline_registry[key],
                    'encoder': cat_encoder
                }

                for fold_data in self.pipeline_registry[key]['split_data']:
                    # for fold_data in data.values():
                    X_cat_train = fold_data['X_train'][cat_cols]
                    X_cat_test = fold_data['X_test'][cat_cols]

                    fold_data['X_train'][cat_cols] = cat_encoder.fit_transform(X_cat_train)
                    fold_data['X_test'][cat_cols] = cat_encoder.transform(X_cat_test)

        self.pipeline_registry = new_pipeline_registry # Atualiza o registro de pipeline com os dados codificados

    def impute_data(self, imputers = []):
        
        new_pipeline_registry = {}

        for key in self.pipeline_registry.keys():
            for imputer in imputers:
                num_imputer = imputer()

                key_imputer = f'imputed_{type(imputer).__name__}_{key}'
                new_pipeline_registry[key_imputer] = {
                    **self.pipeline_registry[key],
                    'imputer': num_imputer
                }

                for data in self.pipeline_registry[key]['split_data']:
                    for fold_data in data.values():
                        X_num_train = fold_data['X_train']
                        X_num_test = fold_data['X_test']

                        fold_data['X_train'] = num_imputer.fit_transform(X_num_train)
                        fold_data['X_test'] = num_imputer.transform(X_num_test)

        self.pipeline_registry = new_pipeline_registry # Atualiza o registro de pipeline com os dados imputados

    def select_model(self, models):

        new_pipeline_registry = {}
        fold_results = []
        summary_rows = []
        original_registry = self.pipeline_registry.copy()

        default_path = Path.cwd() / 'output'
        default_path.mkdir(parents=True, exist_ok=True)
        output_path = default_path

        for model_name, model_info in models.items():
            best_models = {}

            model = model_info['model']

            model = model_info['model']
            hyperparameters = model_info['hyperparameters'] if 'hyperparameters' in model_info else {}
            selection_method = model_info['selection_method'] if 'selection_method' in model_info else 'grid'
            scoring = model_info['scoring'] if 'scoring' in model_info else 'accuracy'
            cv = model_info['cv'] if 'cv' in model_info else 5
            n_iter = model_info['n_iter'] if 'n_iter' in model_info else 10
            random_state = model_info['random_state'] if 'random_state' in model_info else 0

            evaluator = ClassificationEvaluator(output_path)

            for key in original_registry.keys():
                key_model = f'{model_name}_{key}'
                new_pipeline_registry[key_model] = {
                    **original_registry[key],
                    'model': model,
                    'hyperparameters': hyperparameters,
                    'scoring': scoring,
                    'cv': cv
                }
                best_models[key_model] = []

                for data in original_registry[key]['split_data']:
                    X_train, y_train = data['X_train'], data['y_train']
                    X_test, y_test = data['X_test'], data['y_test']

                    X_train.colums = X_train.columns.astype(str)
                    X_test.columns = X_test.columns.astype(str)

                    if selection_method == 'grid':
                        search = GridSearchCV(model, X_train, y_train, hyperparameters=hyperparameters, scoring=scoring, cv=cv)
                    elif selection_method == 'random':
                        search = RandomizedSearchCV(model, X_train, y_train, hyperparameters=hyperparameters, scoring=scoring, cv=cv, n_iter=n_iter, random_state=random_state)
                    # elif selection_method == 'bayes':
                    #     search = BayesSearchCV(model, X_train, y_train, hyperparameters=hyperparameters, scoring=scoring, cv=cv, n_iter=n_iter, random_state=random_state)
                    
                    best_models[key_model].append(search.best_estimator_)

                    y_pred = search.predict(X_test)

                    fold_results.append({'y_true': y_test, 'y_pred': y_pred})

                score = evaluator.evaluate(key_model, fold_results)

                row = {'Key': key_model, 'Accuracy': score}
                summary_rows.append(row)

            self.pipeline_registry = new_pipeline_registry # Atualiza o registro de pipeline com os modelos selecionados

            df_models = pd.DataFrame(summary_rows).round(4)
            main_metric = 'accuracy'
            csv_path = self.output_path / "all_models.csv"
            df_models.to_csv(csv_path, index=False)
            try:
                from IPython.display import display
                print("\n--- Model Selection Summary ---")
                display(df_models)
            except ImportError:
                print("\n--- Model Selection Summary ---")
                print(df_models.to_string())