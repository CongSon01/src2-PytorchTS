'''
from hyperparams import Hyperparams
from train_and_plot_predictions import data_creation, training_model, metrics_rolling_dataset, plot_predictions


test = {'model':'DeepNegPol', 'poids':'forts', 'num_cells1':160 , 'num_cells2':320 ,'epochs':80,'dropout':0.01, 'num_layers':4, 'batch_size':128, 'lr':1e-3}
modele = test['model']
num_cells = 60 if modele == 'DeepNegPol' else test['num_cells']
num_cells1 = test['num_cells1']  if modele == 'DeepNegPol' else 40
num_cells2 = test['num_cells2']  if modele == 'DeepNegPol' else 80
epochs = test['epochs']
dropout = test['dropout']
num_layers = test['num_layers']
batch_size = test['batch_size']
lr = test['lr']
poids = test['poids']
	    
		
params = Hyperparams(modele = modele, learning_rate = lr, epochs = epochs, num_cells1 = num_cells1, 
		                 num_cells2=num_cells2, dropout_rate=dropout, num_layers=num_layers, 
		                 batch_size = batch_size, num_cells = num_cells, poids = poids)
		                 

## DATA
train_ds, test_ds, target_dim = data_creation(params.model_type, params.test_end,params)



#Retrieve model
for rep in range(0, 6):

    from gluonts.model.predictor import Predictor
    from pathlib import Path
    name = str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) if test['model'] == 'DeepNegPol' else str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells:'+str(test['num_cells'])
    		    
    	    
    predictor_deserialized = Predictor.deserialize(Path('save_models/'+name))
    
    predictor = predictor_deserialized
    
    metrics_rolling_dataset("multivariate", test_ds, predictor, params, rep)
    
    #Normal
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal1, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal1")
    	    
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal2, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal2")
    	    
    
    
    #Concert
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert1, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert1")
    	    
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert2, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert2")
    	    
    
    #Events
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event1, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event1")	
    	    
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event2, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event2")		
    	
		    
    #Perturbations
    _, test_ds_pert1, _ = data_creation(params.model_type, params.test_end_pert1,params)
    plot_predictions(test_ds_pert1, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert1")
    		    
    #Perturbation 2
    _, test_ds_pert2, _ = data_creation(params.model_type, params.test_end_pert2,params)
    plot_predictions(test_ds_pert2, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert2")
    	    
    #Perturbation 3
    _, test_ds_pert3, _ = data_creation(params.model_type, params.test_end_pert3,params)
    plot_predictions(test_ds_pert3, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert3")
    	    
    #Perturbation 4
    _, test_ds_pert4, _ = data_creation(params.model_type, params.test_end_pert4,params)
    plot_predictions(test_ds_pert4, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert4")
    	    
    #Perturbation 5
    _, test_ds_pert5, _ = data_creation(params.model_type, params.test_end_pert5,params)
    plot_predictions(test_ds_pert5, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert5")
'''








'''
from hyperparams import Hyperparams
from train_and_plot_predictions import data_creation, training_model, metrics_rolling_dataset, plot_predictions


test = {'model':'LSTMIndScaling', 'poids':'forts', 'num_cells':320 ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}
modele = test['model']
num_cells = 60 if modele == 'DeepNegPol' else test['num_cells']
num_cells1 = test['num_cells1']  if modele == 'DeepNegPol' else 40
num_cells2 = test['num_cells2']  if modele == 'DeepNegPol' else 80
epochs = test['epochs']
dropout = test['dropout']
num_layers = test['num_layers']
batch_size = test['batch_size']
lr = test['lr']
poids = test['poids']
	    
		
params = Hyperparams(modele = modele, learning_rate = lr, epochs = epochs, num_cells1 = num_cells1, 
		                 num_cells2=num_cells2, dropout_rate=dropout, num_layers=num_layers, 
		                 batch_size = batch_size, num_cells = num_cells, poids = poids)
		                 

## DATA
train_ds, test_ds, target_dim = data_creation(params.model_type, params.test_end,params)



#Retrieve model
for rep in range(0, 6):

    from gluonts.model.predictor import Predictor
    from pathlib import Path
    name = str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) if test['model'] == 'DeepNegPol' else str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells:'+str(test['num_cells'])
    		    
    	    
    predictor_deserialized = Predictor.deserialize(Path('save_models/'+name))
    
    predictor = predictor_deserialized
    
    metrics_rolling_dataset("multivariate", test_ds, predictor, params, rep)
    
    #Normal
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal1, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal1")
    	    
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal2, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal2")
    	    
    
    
    #Concert
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert1, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert1")
    	    
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert2, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert2")
    	    
    
    #Events
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event1, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event1")	
    	    
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event2, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event2")		
    	
		    
    #Perturbations
    _, test_ds_pert1, _ = data_creation(params.model_type, params.test_end_pert1,params)
    plot_predictions(test_ds_pert1, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert1")
    		    
    #Perturbation 2
    _, test_ds_pert2, _ = data_creation(params.model_type, params.test_end_pert2,params)
    plot_predictions(test_ds_pert2, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert2")
    	    
    #Perturbation 3
    _, test_ds_pert3, _ = data_creation(params.model_type, params.test_end_pert3,params)
    plot_predictions(test_ds_pert3, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert3")
    	    
    #Perturbation 4
    _, test_ds_pert4, _ = data_creation(params.model_type, params.test_end_pert4,params)
    plot_predictions(test_ds_pert4, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert4")
    	    
    #Perturbation 5
    _, test_ds_pert5, _ = data_creation(params.model_type, params.test_end_pert5,params)
    plot_predictions(test_ds_pert5, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert5")    
    
'''    
    
    
    
    
    
    
    

from hyperparams import Hyperparams
from train_and_plot_predictions import data_creation, training_model, metrics_rolling_dataset, plot_predictions


test = {'model':'GPCOP', 'poids':'forts', 'num_cells':320 ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}
modele = test['model']
num_cells = 60 if modele == 'DeepNegPol' else test['num_cells']
num_cells1 = test['num_cells1']  if modele == 'DeepNegPol' else 40
num_cells2 = test['num_cells2']  if modele == 'DeepNegPol' else 80
epochs = test['epochs']
dropout = test['dropout']
num_layers = test['num_layers']
batch_size = test['batch_size']
lr = test['lr']
poids = test['poids']
	    
		
params = Hyperparams(modele = modele, learning_rate = lr, epochs = epochs, num_cells1 = num_cells1, 
		                 num_cells2=num_cells2, dropout_rate=dropout, num_layers=num_layers, 
		                 batch_size = batch_size, num_cells = num_cells, poids = poids)
		                 

## DATA
train_ds, test_ds, target_dim = data_creation(params.model_type, params.test_end,params)



#Retrieve model
for rep in range(0, 6):

    from gluonts.model.predictor import Predictor
    from pathlib import Path
    name = str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) if test['model'] == 'DeepNegPol' else str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells:'+str(test['num_cells'])
    		    
    	    
    predictor_deserialized = Predictor.deserialize(Path('save_models/'+name))
    
    predictor = predictor_deserialized
    
    metrics_rolling_dataset("multivariate", test_ds, predictor, params, rep)
    
    #Normal
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal1, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal1")
    	    
    _, test_ds_norm, _ = data_creation(params.model_type, params.test_end_normal2, params)
    plot_predictions(test_ds_norm, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Normal2")
    	    
    
    
    #Concert
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert1, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert1")
    	    
    _, test_ds_conc, _ = data_creation(params.model_type, params.test_end_concert2, params)
    plot_predictions(test_ds_conc, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Concert2")
    	    
    
    #Events
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event1, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event1")	
    	    
    _, test_ds_event, _ = data_creation(params.model_type, params.test_end_event2, params)
    plot_predictions(test_ds_event, params.prediction_length, predictor, params.model_type, params.modele, params, rep, period="Event2")		
    	
		    
    #Perturbations
    _, test_ds_pert1, _ = data_creation(params.model_type, params.test_end_pert1,params)
    plot_predictions(test_ds_pert1, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert1")
    		    
    #Perturbation 2
    _, test_ds_pert2, _ = data_creation(params.model_type, params.test_end_pert2,params)
    plot_predictions(test_ds_pert2, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert2")
    	    
    #Perturbation 3
    _, test_ds_pert3, _ = data_creation(params.model_type, params.test_end_pert3,params)
    plot_predictions(test_ds_pert3, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert3")
    	    
    #Perturbation 4
    _, test_ds_pert4, _ = data_creation(params.model_type, params.test_end_pert4,params)
    plot_predictions(test_ds_pert4, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert4")
    	    
    #Perturbation 5
    _, test_ds_pert5, _ = data_creation(params.model_type, params.test_end_pert5,params)
    plot_predictions(test_ds_pert5, params.prediction_length, predictor, params.model_type, params.modele,params, rep, period="Pert5")    	    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    	    
