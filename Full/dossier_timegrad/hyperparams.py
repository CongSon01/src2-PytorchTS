from typing import NamedTuple, List, Optional

# hybridize fails for now
hybridize = False


class Hyperparams(NamedTuple):
    
    #Paramètres à modifier 
    path_folder = "."
    num_cells: int = 40
    learning_rate: float = 1e-3
    learning_rate_fullrank: float = 1e-3
    modele: str =  "LSTMMAF"
    #Pour DeepNegPol seulement
    num_cells1: int = 60
    num_cells2: int = 20
    poids: str =  "forts"

    #Paramètres fixes optimisation
    batch_size: int = 32
    num_batches_per_epoch: int = 80
    epochs: int = 150
    num_eval_samples: int = 100
    num_layers: int = 2
    rank: int = 8
    conditioning_length: int = 50
    target_dim_sample: int = 5
    dropout_rate: float = 0.1
    patience: int = 5
    cell_type: str = "lstm"
    hybridize: bool = True
    minimum_learning_rate: float = 1e-5
    
    
    #Paramètres fixes modele
    pred_days: int = 1
    lags_seq: Optional[List[int]] = [1, 2, 4, 12, 24, 48, 48*7]
    given_days = 3*pred_days
    model_type = "univariate" if modele == "DeepAR" else "multivariate"
    
    
    #Paramètres fixes donnees
    train_start = '2019-04-01 01:00'
    train_end = '2022-01-31 00:30'
    test_start = '2022-02-01 01:00'
    test_end = '2022-04-11 00:30'
    freq='30min'
    prediction_length=48*pred_days
    
    #Jours normaux
    test_end_normal1 = '2022-05-19 00:30'
    test_end_normal2 = '2022-05-21 00:30'
    
    #Jours particuliers
    test_end_concert1 = '2022-03-16 00:30'
    test_end_concert2 = '2022-05-14 00:30'
    
    test_end_event1 = '2022-04-02 00:30'
    test_end_event2 = '2022-05-08 00:30'
    
    test_end_pert1 = '2022-04-04 00:30'  #Grosse pert, aug RER -> M1
    test_end_pert2 = '2022-03-02 00:30' #Pert, aug M1 -> RER
    test_end_pert3 = '2022-04-06 00:30' #Petite pert
    test_end_pert4 = '2022-05-20 00:30' 
    test_end_pert5 = '2022-03-01 00:30'     


    
    
    

class FastHyperparams(NamedTuple):
    p = Hyperparams()
    epochs: int = 1
    num_batches_per_epoch: int = 1
    num_cells: int = 1
    num_layers: int = 1
    num_eval_samples: int = 1
    modele: str = "DeepNegPol"
    cell_type: str = "lstm"
    poids: str =  "forts"
    conditioning_length: int = 10
    batch_size: int = 16
    rank: int = 5

    target_dim_sample: int = p.target_dim_sample
    patience: int = p.patience
    hybridize: bool = hybridize
    learning_rate: float = p.learning_rate
    learning_rate_fullrank: float = p.learning_rate_fullrank
    minimum_learning_rate: float = p.minimum_learning_rate
    dropout_rate: float = p.dropout_rate
    lags_seq: Optional[List[int]] = p.lags_seq
    #scaling: bool = p.scaling


if __name__ == '__main__':
    params = Hyperparams()
    print(repr(params))

