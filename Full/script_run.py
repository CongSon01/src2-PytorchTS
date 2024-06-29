


# Standard library imports
def train_mod(test, rep):
 
    try:    
	    from hyperparams import Hyperparams
	    from train_and_plot_predictions import data_creation, training_model, metrics_rolling_dataset, plot_predictions


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
		                 
	    ##############################################################################################
	    # SAVE THE DATA
	    ##############################################################################################
	    import dataset
	    dataset.create_data_files(poids)


	    ##############################################################################################
	    # IMPORTATION DONNEES - COMPTAGES + COVARIABLES
	    ##############################################################################################
	    train_ds, test_ds, target_dim = data_creation(params.model_type, params.test_end,params)
	    

	    ##############################################################################################
	    # ENTRAINEMENT
	    ###########################################################################################
	    predictor = training_model(params.modele, train_ds, target_dim, params = params)
	    
	    #save model
	    name = str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) if test['model'] == 'DeepNegPol' else str(test['model']) + str(rep) + 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells:'+str(test['num_cells'])
		    
	    os.makedirs('save_models/'+name, exist_ok=True)
	    predictor.serialize(pathlib.Path('save_models/'+name))
		    
	    ##############################################################################################
	    # CALCULATE METRICS ON ROLLING DATASET
	    ##############################################################################################
	    metrics_rolling_dataset(params.model_type, test_ds, predictor, params, rep)
		    
		    
	    ##############################################################################################
	    #PLOTS FROM PARTICULAR PERIODS
	    ##############################################################################################
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
	    
    
    except:
    	    print("An exception occurred") 
	 


#test0 = {'model':'LSTMIndScaling', 'poids':'forts', 'num_cells':320  ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}
#test1 = {'model':'DeepNegPol', 'poids':'forts', 'num_cells1':160 , 'num_cells2':320 ,'epochs':80,'dropout':0.01, 'num_layers':4, 'batch_size':128, 'lr':1e-3}
#test2 = {'model':'LSTMMAF', 'poids':'forts', 'num_cells':160  ,'epochs':80,'dropout':0.01, 'num_layers':4, 'batch_size':128, 'lr':1e-3}
#test2 = {'model':'GPCOP', 'poids':'forts', 'num_cells':320 ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}
test3 = {'model':'GPScaling', 'poids':'forts', 'num_cells':320  ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}
#test3 = {'model':'LSTMCOP', 'poids':'forts', 'num_cells':320  ,'epochs':80,'dropout':0.01, 'num_layers':3, 'batch_size':128, 'lr':1e-3}


list_tests_str = ['test' + str(i) for i in range (0, 25)]
list_tests= []
for test in list_tests_str:
    if test in locals():
        list_tests.append(eval(test))




import os
import pathlib
for test in list_tests:

    for rep in range(0, 6):
    
    	name =  'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells1:'+str(test['num_cells1'])+'nbcells2:'+str(test['num_cells2']) if test['model'] == 'DeepNegPol' else 'poids:'+str(test['poids'])+'_' + 'lr:'+str(test['lr']) +'_'+'n_layers:'+str(test['num_layers']) +'_'+'epochs:'+str(test['epochs']) +'_'+ 'dropout:'+str(test['dropout']) +'_'+'lr:'+str(test['lr'])+'_'+'nbcells:'+str(test['num_cells'])
    
    
    	if os.path.isfile('./results/'+ str(rep) +str(test['model'])+'_results/metrics_'+name+'.csv') == False:
    		train_mod(test, rep)
    		
    	else:
    		print("Model deja entrainé")

