# Standard library imports

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from gluonts.evaluation import MultivariateEvaluator, Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from hyperparams import Hyperparams
from multivariate_models import models_dict

from gluonts.dataset.common import ListDataset

import warnings
warnings.filterwarnings("ignore")

import os

from gluonts.dataset.rolling_dataset import (
    StepStrategy,
    generate_rolling_dataset,
)
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.util import to_pandas


params = Hyperparams(hybridize=True)
path_folder = params.path_folder 



#############################################################################################
# IMPORTATION DONNEES - COMPTAGES + COVARIABLES
##############################################################################################

def data_creation(model_type, test_end,params):
    #importation data
    data_frame = pd.read_csv('Count_data.csv', sep=",", index_col=0, parse_dates=True, decimal='.')
    covariates = pd.read_csv('Covariates.csv', sep=",", index_col=0, parse_dates=True, decimal='.')
    weight_indic = pd.read_csv('weight_indic.csv', sep=",", index_col=0, parse_dates=True, decimal='.')
    target_dim = data_frame.shape[1]

    params = params

    train_start = pd.Timestamp(params.train_start , freq='30T')
    train_end = pd.Timestamp(params.train_end , freq='30T')
    test_start = pd.Timestamp(params.test_start , freq='30T')
    test_end = pd.Timestamp(test_end , freq='30T')
    
    
    freq = params.freq

    pred_days = params.pred_days

    #Pour données multivariées
    if model_type == 'multivariate':
        """
        covariates_train = covariates[:train_end]
        covariates_train = covariates_train.transpose()

        covariates_test = covariates[test_start:test_end]
        covariates_test = covariates_test.transpose()

        data_frame_train = data_frame[:train_end]
        data_frame_train = data_frame_train.transpose()

        data_frame_test = data_frame[test_start:test_end]
        data_frame_test = data_frame_test.transpose()
        
        weight_indic_train = weight_indic[:train_end]
        weight_indic_train = weight_indic_train.transpose()
        
        weight_indic_test = weight_indic[test_start:test_end]
        weight_indic_test = weight_indic_test.transpose()
        
        #Création de la valeur de scaling
        datatot = data_frame_train.transpose().sum(axis=1)
        nonzero_sumT = (datatot!=0).sum()
        vT_input = np.true_divide(datatot.sum(),nonzero_sumT)+1
        np.save(path_folder+'/vT', vT_input)
        """
        covariates_train = covariates[:train_end]
        covariates_train = covariates_train.transpose()

        covariates_test = covariates[:test_end]
        covariates_test = covariates_test.transpose()

        data_frame_train = data_frame[:train_end]
        data_frame_train = data_frame_train.transpose()

        data_frame_test = data_frame[:test_end]
        data_frame_test = data_frame_test.transpose()
        
        weight_indic_train = weight_indic[:train_end]
        weight_indic_train = weight_indic_train.transpose()
        
        weight_indic_test = weight_indic[:test_end]
        weight_indic_test = weight_indic_test.transpose()
        
        #Création de la valeur de scaling
        #datatot = data_frame_train.transpose().sum(axis=1)
        #nonzero_sumT = (datatot!=0).sum()
        #vT_input = np.true_divide(datatot.sum(),nonzero_sumT)+1
        #np.save(path_folder+'/vT', vT_input)


        train_ds = ListDataset([{"start": train_start, "target": data_frame_train, "feat_dynamic_real":covariates_train, "feat_dynamic_cat":weight_indic_train}], freq=freq, one_dim_target=False)
        test_ds = ListDataset([{"start": test_start, "target": data_frame_test, "feat_dynamic_real":covariates_test, "feat_dynamic_cat":weight_indic_test}], freq=freq, one_dim_target=False)
        
    #Pour données univariées
    elif model_type == 'univariate':
        covariates_train = covariates[:train_end]
        covariates_train = covariates_train.transpose().to_numpy()
        covariates_train_l = [covariates_train]*data_frame.shape[1]

        covariates_test = covariates[test_start:test_end]
        covariates_test = covariates_test.transpose().to_numpy()
        covariates_test_l = [covariates_test]*data_frame.shape[1]


        weight_indic_train = weight_indic[:train_end]
        weight_indic_train = weight_indic_train.transpose().to_numpy()
        weight_indic_train_l = [weight_indic_train]*data_frame.shape[1]
        
        
        weight_indic_test = weight_indic[test_start:test_end]
        weight_indic_test = weight_indic_test.transpose().to_numpy()
        weight_indic_test_l = [weight_indic_test]*data_frame.shape[1]
        

        data_frame_train = data_frame[:train_end]
        data_frame_train = data_frame_train.transpose().to_numpy()

        data_frame_test = data_frame[test_start:test_end]
        data_frame_test = data_frame_test.transpose().to_numpy()

        series_ids = (data_frame.transpose().reset_index())['index'].astype('category').cat.codes.values.T
        series_ids = np.expand_dims(series_ids, axis=1)
        series_cat_cardinalitie = len(series_ids)


        prediction_length=48*pred_days

        dates_train = [train_start for _ in range(series_cat_cardinalitie)]
        dates_test = [test_start for _ in range(series_cat_cardinalitie)]


        train_ds = ListDataset([
            {
                "target": target,
                "start": start,
                "feat_dynamic_real": fdr,
                "feat_static_cat": fsc,
                "feat_dynamic_cat": fdc
            }
            for (target, start, fdr, fsc, fdc) in zip(data_frame_train,
                                                 dates_train,
                                                 covariates_train_l,
                                                 series_ids,
                                                 weight_indic_train_l)
        ], freq="30min")



        test_ds = ListDataset([
            {
                "target": target,
                "start": start,
                "feat_dynamic_real": fdr,
                "feat_static_cat": fsc,
                "feat_dynamic_cat": fdc
            }
            for (target, start, fdr, fsc, fdc) in zip(data_frame_test,
                                                 dates_test,
                                                 covariates_test_l,
                                                 series_ids,
                                                 weight_indic_train_l)
        ], freq="30min")
    
    
    return train_ds, test_ds, target_dim



##############################################################################################
# ENTRAINEMENT
##############################################################################################
def training_model(modele, train_ds, target_dim, params):
    params = params

    from multivariate_models import models_dict
    estimator = models_dict[modele](
         freq="30min",
         prediction_length=params.prediction_length,
         target_dim=target_dim,
         params=params,
         )


    predictor = estimator.train(train_ds)

    return predictor


##############################################################################################
# CALCULATE METRICS ON ROLLING DATASET
##############################################################################################

def truncate_features(timeseries: dict, max_len: int) -> dict:
    for key in (
        FieldName.FEAT_DYNAMIC_REAL,
    ):
        if key not in timeseries:
            continue
        timeseries[key] = (timeseries[key])[:,:max_len]

    return timeseries

def metrics_rolling_dataset(model_type,test_ds, predictor,params, rep):
    
    params = params

    strategy=StepStrategy(
            prediction_length=params.prediction_length,
            step_size=48
        )
    start_time = pd.Timestamp(params.test_start, freq='30T')
    end_time = pd.Timestamp(params.test_end, freq='30T')
    
    if model_type == "multivariate":
        #create rolling data
        ds = []

        item = (next(iter(test_ds)))
        target = item["target"]
        start = item["start"]

        index = pd.date_range(start=start, periods=target.shape[1], freq=params.freq)
        series = pd.DataFrame(target.T, index=index)

        base = series[:start_time][:-1].to_numpy()
        prediction_window = series[start_time:end_time]
        nb_j = 0
        
        for window in strategy.get_windows(prediction_window):
            nb_j = nb_j + 1
            new_item = item.copy()
            new_item[FieldName.TARGET] = np.concatenate(
                [base, window.to_numpy()]
            ).T
            new_item = truncate_features(
                new_item, new_item[FieldName.TARGET].shape[1]
            )
            ds.append(new_item)
            
            if nb_j > 30:
                break

        #forecast rolling data
        forecast_it, ts_it = make_evaluation_predictions(
                ds, predictor=predictor, num_samples=params.num_eval_samples
            )


        print("predicting")
        forecasts = list(forecast_it)
        targets = list(ts_it)


    
        # evaluate
        evaluator = MultivariateEvaluator(
                    quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum}
                )

        agg_metrics, item_metrics = evaluator(
                    targets, forecasts
                )

        print("\nCRPS:", agg_metrics["mean_wQuantileLoss"])
        print("\nCRPS-sum:", agg_metrics["m_sum_mean_wQuantileLoss"])    
        print("\nND:", agg_metrics["ND"])    
        
        
    if model_type == "univariate":
        dataset_rolled = generate_rolling_dataset(
            dataset=test_ds,
            start_time=start_time,
            end_time=end_time,
            strategy=strategy
            )
        

        ds = []
        for item in test_ds:
            series = to_pandas(item, start_time.freq)
            base = series[:start_time][:-1].to_numpy()
            prediction_window = series[start_time:end_time]
            nb_j = 0

            for window in strategy.get_windows(prediction_window):
                nb_j = nb_j + 1
                new_item = item.copy()
                new_item[FieldName.TARGET] = np.concatenate(
                [base, window.to_numpy()]
                )
                new_item = truncate_features(
                new_item, len(new_item[FieldName.TARGET])
                )
                ds.append(new_item)
        
                if nb_j > 8:
                    break
        
        
        forecast_it, ts_it = make_evaluation_predictions(
            ds, predictor=predictor, num_samples=params.num_eval_samples
        )


        #print("predicting")
        forecasts = list(forecast_it)
        targets = list(ts_it)
        
        evaluator = Evaluator(
                quantiles=(np.arange(20) / 20.0)[1:]
            )

        agg_metrics, item_metrics = evaluator(
                targets, forecasts
            )

        #print(agg_metrics)
        print("\nCRPS:", agg_metrics["mean_wQuantileLoss"])
        print("\nND:", agg_metrics["ND"])
        
    
    modele = params.modele
    lr=params.learning_rate_fullrank if modele in ['LSTMFRScaling', 'LSTMFR'] else params.learning_rate
    name =  str(rep)+'_'+'poids:'+str(params.poids)+'_' + 'lr:'+str(params.learning_rate) +'_'+'n_layers:'+str(params.num_layers) +'_'+'epochs:'+str(params.epochs) +'_'+ 'dropout:'+str(params.dropout_rate) +'_'+'lr:'+str(lr)+'_'+'nbcells1:'+str(params.num_cells1)+'nbcells2:'+str(params.num_cells2) if modele == 'DeepNegPol' else str(rep)+'_'+'poids:'+str(params.poids)+'_' + 'lr:'+str(params.learning_rate) +'_'+'n_layers:'+str(params.num_layers) +'_'+'epochs:'+str(params.epochs) +'_'+ 'dropout:'+str(params.dropout_rate) +'_'+'lr:'+str(lr)+'_'+'nbcells:'+str(params.num_cells)

    metrics = pd.DataFrame.from_records(agg_metrics, index=[modele]).transpose()
    metrics['Settings'] = name
    plot_log_path = path_folder+"/results/"+modele+"_results/"
    directory = os.path.dirname(plot_log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    metrics.to_csv('{}metrics_{}.csv'.format(plot_log_path,name))
    print('Done')


##############################################################################################
#PLOTS FROM PARTICULAR PERIODS
##############################################################################################
locations = ['P1_S', 'P1_E', 'P11_E', 'P11_S', 'P7_S', 'P7_E', 'P12_S', 'P12_E','P2_S','P2_E',
             'P10_S','P10_E', 'P5_E', 'P5_S', 'P6_E','P6_S', 'P9_E', 'P9_S', 'METRO_F6','RER_F1',
             'RER_F2','RER_F5', 'METRO_F1','RER_F3', 'RER_F4'
             ]

def plot_multivariate(target, forecast, prediction_length):
    rows = 9
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(40, 60))
    axx = axs.ravel()
    seq_len, target_dim = target.shape
    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        target[-prediction_length :][dim].plot(ax=ax)

        # (quantile, target_dim, seq_len)
        pred_df = pd.DataFrame(
            {q: forecast.quantile(q)[:,dim] for q in [0.05, 0.5, 0.95]},
            index=forecast.index,
        )

        ax.fill_between(
            forecast.index, pred_df[0.05], pred_df[0.95], alpha=0.4, color='r'
        )
        #pred_df[0.5].plot(ax=ax, color='g')
        pred_df[0.95].plot(ax=ax, color='r')
        ax.set_title(label=locations[dim], fontdict={'fontsize': 24, 'fontweight': 'medium'} )
        ax.xaxis.label.set_size(30)
        ax.yaxis.label.set_size(30)
        ax.yaxis.set_tick_params(labelsize=25)
        ax.xaxis.set_tick_params(labelsize=0)
    #plt.show()
    
    return fig
    
    
def plot_univariate(target, forecast, prediction_length):
    rows = 9
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(6, 12))
    axx = axs.ravel()
    seq_len = target[0].shape[0]
    target_dim = len(target)
    for dim in range(0, min(rows * cols, target_dim)):
        ax = axx[dim]

        (target[dim])[-3 * prediction_length :].plot(ax=ax)

        # (quantile, target_dim, seq_len)
        pred_df = pd.DataFrame(
            {q: (forecast[dim]).quantile(q) for q in [0.1, 0.5, 0.9]},
            index=(forecast[dim]).index,
        )

        ax.fill_between(
            (forecast[dim]).index, pred_df[0.1], pred_df[0.9], alpha=0.2, color='g'
        )
        pred_df[0.5].plot(ax=ax, color='g')
        
        ax.set_title(label=locations[dim])
    #plt.show()
    
    return fig



def plot_predictions(test_ds, prediction_length, predictor, model_type, modele,params, rep, period="Concert"):
    

    params = params

    forecast_it, ts_it = make_evaluation_predictions(
        test_ds, predictor=predictor, num_samples=params.num_eval_samples
    )


    forecasts = list(forecast_it)
    targets = list(ts_it)

    if model_type == "multivariate":
        fig = plot_multivariate(
            target=targets[0],
            forecast=forecasts[0],
            prediction_length=prediction_length,
            )
   
        
        
        
    if model_type == "univariate":
        fig = plot_univariate(
            target=targets,
            forecast=forecasts,
            prediction_length=prediction_length,
            )


    lr=params.learning_rate_fullrank if modele in ['LSTMFRScaling', 'LSTMFR'] else params.learning_rate
    name =  str(rep)+'_'+'poids:'+str(params.poids)+'_' + 'lr:'+str(params.learning_rate) +'_'+'n_layers:'+str(params.num_layers) +'_'+'epochs:'+str(params.epochs) +'_'+ 'dropout:'+str(params.dropout_rate) +'_'+'lr:'+str(lr)+'_'+'nbcells1:'+str(params.num_cells1)+'nbcells2:'+str(params.num_cells2) if modele == 'DeepNegPol' else str(rep)+'_'+'poids:'+str(params.poids)+'_' + 'lr:'+str(params.learning_rate) +'_'+'n_layers:'+str(params.num_layers) +'_'+'epochs:'+str(params.epochs) +'_'+ 'dropout:'+str(params.dropout_rate) +'_'+'lr:'+str(lr)+'_'+'nbcells:'+str(params.num_cells)

    plot_log_path = path_folder+"/results/"+modele+"_results/"
    directory = os.path.dirname(plot_log_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


    fig.savefig('{}plot_{}.jpg'.format(plot_log_path, name+'_'+period))
    plt.close()
