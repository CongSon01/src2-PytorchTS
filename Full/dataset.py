import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from hyperparams import Hyperparams



##############################################################################################
# DATA
##############################################################################################

locations = ['P1_S', 'P1_E', 'P11_E', 'P11_S', 'P7_S', 'P7_E', 'P12_S', 'P12_E','P2_S','P2_E',
             'P10_S','P10_E', 'P5_E', 'P5_S', 'P6_E','P6_S', 'P9_E', 'P9_S', 'METRO_F6','RER_F1',
             'RER_F2','RER_F5', 'METRO_F1','RER_F3', 'RER_F4'
             ]
name = 'LD_flux.txt'
save_name = 'LD'

##COUNT DATA
data_frame = pd.read_csv('data/LD/LD_flux.txt', sep=",", index_col=0, parse_dates=True, decimal='.')[['date','hr'] + locations]
datetime_format = data_frame['date'] + ' '  + data_frame['hr'].map(int).replace(to_replace=24, value=0).map(str).apply(lambda x: '{0:0>2}'.format(x)) + ':' + np.round(((data_frame['hr'] - data_frame['hr'].map(int))*60),0).map(int).map(str).apply(lambda x: '{0:0>2}'.format(x))
data_frame = data_frame.set_index(pd.to_datetime(datetime_format).dt.strftime('%Y-%m-%d %H:%M'))
data_frame.index.name = 'time'
data_frame = data_frame.drop(['date','hr'],axis=1)
data_frame.fillna(0, inplace=True)
#Đọc code: tạo file Count data -> Đã xong   
#data_frame = data_frame + 1




#COVARIATES
def gen_covariates(poids, times, num_covariates):
    if poids == "forts": #Chưa hiểu sâu nhưng có vẻ không cần dùng cho dữ liệu của mình ?
    	w_poids = [20000, 10000, 100] #Chưa hiểu sâu nhưng có vẻ không cần dùng cho dữ liệu của mình ?
    if poids == "faibles": #Chưa hiểu sâu nhưng có vẻ không cần dùng cho dữ liệu của mình ?
    	w_poids = [2000, 1000, 10] #Chưa hiểu sâu nhưng có vẻ không cần dùng cho dữ liệu của mình ?

    covariates = np.zeros((times.shape[0], num_covariates))
    weights = np.zeros((times.shape[0], num_covariates-26)) #Trừ 26 là vì mấy cái thời gian (calendar factors) không có trọng số
    day_of_week = np.zeros((times.shape[0]))
    
    times_df = times.to_frame(index=False)
    times_df['time'] = pd.to_datetime(times_df['time'])
    times_df['date'] = times_df['time'].astype(str).str[:10]
    times_df['hr'] =  round((times_df['time'].dt.hour.replace(to_replace=0, value=24) + times_df['time'].dt.minute/60),1)
    
    #public hollidays and hollidays
    data_holidays = pd.read_csv('data/LD/fr-en-calendrier-scolaire.csv', sep=";", parse_dates=True, decimal='.')
    data_holidays = data_holidays[data_holidays['Académies']=="Paris"]
    data_holidays = data_holidays[data_holidays['Population']!="Enseignants"]
    data_holidays = data_holidays[data_holidays['annee_scolaire'].str[:4].astype(float) >= 2018]
    data_holidays['Date de début']= pd.to_datetime(data_holidays['Date de début'].astype(str).str[:10])
    data_holidays['Date de fin']= pd.to_datetime(data_holidays['Date de fin'].astype(str).str[:10])
    
   # data_ferie = pd.read_csv('data/LD/jours_feries_metropole.csv', sep=",", parse_dates=True, decimal='.')
    
    holiday_dates = []
    for i in range(len(data_holidays)):
        holiday_dates.append(date_range_list(data_holidays['Date de début'].iloc[i],data_holidays['Date de fin'].iloc[i]))
    holiday_dates = [inner for outer in holiday_dates for inner in outer]
    holiday_df = pd.DataFrame (holiday_dates, columns=['date_hol']).astype(str)

    
    day_type = pd.merge(times_df, holiday_df, how='left', left_on = 'date', right_on = 'date_hol')
    day_type['hol'] = np.where(day_type['date_hol'].isna(), 0, 1)
    day_type = pd.merge(day_type, data_ferie, how='left', left_on = 'date', right_on = 'date')
    day_type['fer'] = np.where(day_type['nom_jour_ferie'].isna(), 0, 1)

    
    for i, input_time in enumerate(times):
        #Chuyển đổi thành định dạng date time
        input_time = datetime.strptime(input_time, '%Y-%m-%d %H:%M')
        #Trích ra ngày trong năm
        day_of_year = input_time.timetuple().tm_yday
        #Tiến hành mã hóa ngày trong năm
        covariates[i, 1:9] = encode(day_of_year, 365)

        hour = input_time.hour #trích giờ ra
        if hour == 0: 
            hour = 24
        timestep = (hour + input_time.minute/60)*2 -1  
        covariates[i, 9:17] = encode(timestep, 48) #Encode POSITON OF THE DAY
        day_of_week[i] = input_time.weekday() 
#Mảng day_of_week sẽ chứa các số nguyên từ 0 đến 6, tương ứng với các ngày trong tuần theo thứ tự: Thứ Hai = 0, Thứ Ba = 1, ..., Chủ Nhật = 6.
        
    covariates[:, 17:24] = pd.get_dummies(pd.Series(day_of_week)).to_numpy() #One hot encode cho ngày trong tuần
    
    covariates[:, 24:26] = day_type[['hol','fer']].to_numpy() #holiday
    
    #ARENA concerts et événements sportifs không liên quan
    data_arena = (pd.read_csv('data/LD/arena.csv', sep=",", parse_dates=True, decimal='.'))[['date', 'hr','counter_ConAv','counter_ConAp','counter_EvAv','counter_EvAp']]   
    data_arena['counter_ConAv'] = np.where(data_arena['counter_ConAv']==0, np.nan, data_arena['counter_ConAv'])
    data_arena['counter_ConAp'] = np.where(data_arena['counter_ConAp']==0, np.nan, data_arena['counter_ConAp'])
    data_arena['counter_EvAv'] = np.where(data_arena['counter_EvAv']==0, np.nan, data_arena['counter_EvAv'])
    data_arena['counter_EvAp'] = np.where(data_arena['counter_EvAp']==0, np.nan, data_arena['counter_EvAp'])    
    
    times_df = pd.merge(times_df, data_arena, how='left', left_on = ['date', 'hr'], right_on = ['date', 'hr'])
    
    w_arena = np.nan_to_num(times_df[['counter_ConAv','counter_ConAp','counter_EvAv','counter_EvAp']].to_numpy())
   # w_arena = 1/(((w_arena != 0)*1).sum(axis=0)/w_arena.shape[0]) * ((w_arena != 0)*1)
    w_arena = np.array((w_poids[0], w_poids[0], w_poids[1], w_poids[1])) * ((w_arena != 0)*1)
    weights[:,0:4] = w_arena
    
    times_df['counter_ConAv'] = np.where(times_df['counter_ConAv'].isna(), 0, times_df['counter_ConAv'])
    times_df['counter_ConAp'] = np.where(times_df['counter_ConAp'].isna(), 0, times_df['counter_ConAp'])   
    times_df['counter_EvAv'] = np.where(times_df['counter_EvAv'].isna(), 0, times_df['counter_EvAv'])   
    times_df['counter_EvAp'] = np.where(times_df['counter_EvAp'].isna(), 0, times_df['counter_EvAp'])   



    times_df = pd.get_dummies(data=times_df, columns=['counter_ConAv', 'counter_ConAp','counter_EvAv','counter_EvAp'], drop_first=True)


    conc_cols = [col for col in times_df.columns for col2 in ['counter_ConAv', 'counter_ConAp','counter_EvAv','counter_EvAp'] if col2 in col]

    #covariates[:, 40:48] = times_df[['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc']].to_numpy()    
    covariates[:, 26:46] = times_df[conc_cols].to_numpy()
    
    #TRAVAUX RER

    data_travaux_RER = (pd.read_csv('data/LD/travaux_rer.csv', sep=",", parse_dates=True, decimal='.'))[['date', 'hr','zone']]  
    data_travaux_RER['value'] = 1
    data_travaux_RER_d = data_travaux_RER.pivot_table(index=['date','hr'],columns='zone',values='value').reset_index(level=['date', 'hr'])
    times_df = pd.merge(times_df, data_travaux_RER_d, how='left', left_on = ['date', 'hr'], right_on = ['date', 'hr'])
    for zone in data_travaux_RER_d.columns[2:].tolist():
        times_df[zone] = np.where(times_df[zone].isna(), 0, times_df[zone])   
    covariates[:, 46:54] = times_df[data_travaux_RER_d.columns[2:].tolist()].to_numpy()
    
    w_travaux = times_df[data_travaux_RER_d.columns[2:].tolist()].to_numpy()
    w_travaux = 1/(((w_travaux != 0)*1).sum(axis=0)/w_travaux.shape[0]) * ((w_travaux != 0)*1)
    weights[:,4:12] = w_travaux
    
    
    
    #METEO
    #Trích dữ liệu thời tiết, chỉ lấy một số cột mà tác giả quan tâm
    meteo = (pd.read_csv('data/LD/meteo.csv', sep=",", parse_dates=True, decimal='.'))[['date', 'hr','temperature','precipitations3h']]  
    #Thêm vào bảng times_df
    times_df = pd.merge(times_df, meteo, how='left', left_on = ['date', 'hr'], right_on = ['date', 'hr'])
    #Kiểm tra xem có NAN trong 2 cột "temperature" và "precipitations3h" hay không? Nếu có thì thay bằng 0
    for met in ['temperature','precipitations3h']:
        times_df[met] = np.where(times_df[met].isna(), 0, times_df[met])   
    #Đưa vào bảng covariates và chuẩn hóa
    covariates[:, 54:56] = times_df[['temperature', 'precipitations3h']].to_numpy()
    covariates[:,54:56] = covariates[:,54:56]/(covariates[:, 54:56]).max(0)
    #Tính trọng số
    w_meteo = np.absolute(times_df[['temperature', 'precipitations3h']].to_numpy())
    weights[:,12:14] = w_meteo
    
    
    #OFFRE RER 
    #On considère ici le nombre de passages de RER - la médiane au niveau de stations situées en amont et aval de La Défense pour tenter d'anticiper les flux à la Défense à t+1.
    # On aura entre autres :
        # type 1 : Nanterre Pref (branche cergy) vers LD
        # type 2 : Nanterre Pref (branche cergy) depuis LD
        # type 3 : Nanterre ville vers LD
        # type4 : Nanterre ville depuis LD
        # type 5 : Bry Marne vers LD
        # type 6 : Bry Marne depuis LD
        # type 7 : Nogent Marne vers LD
        #type 8 : Nogent Marne depuis LD
    data_offre = (pd.read_csv('data/LD/tt_rera_Q.csv', sep=",", parse_dates=True, decimal='.'))[['date', 'hr','name','value']] 
    data_offre['hr'] = np.round(data_offre['hr'], 1)
    data_offre = data_offre.rename(columns={"name": "type_passage", "value": "diff"})
    
    #on évite de considérer les périodes de grève et de covid
    data_offre['diff'] = np.where(( (data_offre['date'] == '2021-01-24') | (data_offre['date'] == '2021-08-23') | (data_offre['date'] == '2019-09-13') | (data_offre['date'] >= '2019-12-05') & (data_offre['date'] <= '2020-01-16')) | ((data_offre['date'] >= '2020-03-15') & (data_offre['date'] <= '2020-06-11')) | ((data_offre['date'] >= '2020-10-29') & (data_offre['date'] <= '2020-12-15')), 1, data_offre['diff'])
    #on considère que les périodes de travaux "camouflent" l'effet des offres rer des branches concernées
    data_link_trav_offre = {'Aub_LD_2020': ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'],
                            'Aub_LD_2021': ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'],
                            'Aub_NantU': ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'],
                            'Aub_Vinc': ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'],
                            'Nant_CerPoi': ['arrivee_ON','depart_ON'],
                            'Reuil_SG' : ['arrivee_OS','depart_OS'],
                            'Sartrou_Cergy' : ['arrivee_ON','depart_ON'],
                            'Sartrou_Poissy' : ['arrivee_ON','depart_ON']}
    data_link_trav_offre =  pd.concat({k: pd.DataFrame({'zone':k,'type_passage':v}) for k, v in data_link_trav_offre.items()}).reset_index(drop=True)
    data_travaux_offre = (pd.merge(data_travaux_RER, data_link_trav_offre, how='left', left_on = ['zone'], right_on = ['zone']))[['date','hr','type_passage','value']]
    data_offre = pd.merge(data_offre, data_travaux_offre, how='left', left_on = ['date','hr','type_passage'], right_on = ['date','hr','type_passage'])
    data_offre['diff'] = np.where(data_offre['value'] == 1, 1, data_offre['diff'])
    data_offre = data_offre.groupby(['date','hr','type_passage'])['diff'].max()
    data_offre = pd.DataFrame(data_offre).reset_index(level=['date', 'hr','type_passage'])
    #data_offre['diff'] = np.exp(-data_offre['diff']) - 1

    #data_offre['diff'] = np.where(data_offre['diff']<0.7,0,data_offre['diff'])

    data_offre = data_offre.pivot_table(index=['date','hr'],columns='type_passage',values='diff').reset_index(level=['date', 'hr']) 
    
    #création des 7 catégories
    #data_offre['double_pert'] = np.where(((data_offre['type9'] < .7) & (data_offre['type10'] < 1)) | ((data_offre['type10'] < .7) & (data_offre['type9'] < 1)) , 1.0, 0)
    #data_offre['arrivee_ON'] = np.where( (data_offre['double_pert'] == 0) & (data_offre['type1'] < .7)  , 1.0, 0)
    #data_offre['arrivee_OS'] = np.where( (data_offre['double_pert'] == 0) & (data_offre['type3'] < .7)  , 1.0, 0)
    #data_offre['arrivee_EST'] = np.where( (data_offre['double_pert'] == 0) & ((data_offre['type5'] < .7) | (data_offre['type7'] < .7) ) , 1.0, 0)
    #data_offre['depart_ON'] = np.where( (data_offre['double_pert'] == 0) & (data_offre['type2'] < .7)  , 1.0, 0)   
    #data_offre['depart_OS'] = np.where( (data_offre['double_pert'] == 0) & (data_offre['type4'] < .7)  , 1.0, 0)
    #data_offre['depart_EST'] = np.where( (data_offre['double_pert'] == 0) & ((data_offre['type6'] < .7) | (data_offre['type8'] < .7) ) , 1.0, 0)
    
    
    #data_offre = data_offre[['date', 'hr','double_pert','arrivee_ON','arrivee_OS','arrivee_EST','depart_ON','depart_OS','depart_EST']]
    #data_offre = data_offre[['date', 'hr','type1','type2','type3','type4','type5','type6','type7','type8']]

    
    #on traite ensuite l'offre sur la table retravaillée
    times_df = pd.merge(times_df, data_offre, how='left', left_on = ['date', 'hr'], right_on = ['date', 'hr'])
    for type in ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc']:
        times_df[type] = np.where(times_df[type].isna(), 1, times_df[type])   


    w_offre = (times_df[['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc']].to_numpy() ) -1
    #w_offre[w_offre != 0] = 10
    
    #w_offre = 1/(((w_offre > 2)*1).sum(axis=0)/w_offre.shape[0]) * (w_offre)**2    
    weights[:,14:22] = w_offre*w_poids[2]


    times_df = pd.get_dummies(data=times_df, columns=['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'], drop_first=True)


    pert_cols = [col for col in times_df.columns for col2 in ['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc'] if col2 in col]

    #covariates[:, 40:48] = times_df[['arrivee_EST1', 'arrivee_EST2','arrivee_ON','arrivee_OS','depart_EST','depart_ON','depart_OS','arrivee_ESTtc']].to_numpy()
    covariates[:, 56:96] = ((times_df[pert_cols].to_numpy())[:,:]) 




    return covariates[:, :num_covariates], weights



#Đã hiểu - có liên quan
def date_range_list(start_date, end_date):
    # Return list of datetime.date objects between start_date and end_date (inclusive).
    date_list = []
    curr_date = start_date
    while curr_date <= end_date:
        date_list.append(curr_date)
        curr_date += timedelta(days=1)
    return date_list




#Đã hiểu - có liên quan
def encode(day_of_year, max_val):
    sin1 = np.sin(2  * np.pi * day_of_year/max_val)
    sin2 = np.sin(4  * np.pi * day_of_year/max_val)
    sin3 = np.sin(6  * np.pi * day_of_year/max_val)
    sin4 = np.sin(8  * np.pi * day_of_year/max_val)
    
    cos1 = np.cos(2 * np.pi * day_of_year/max_val)
    cos2 = np.cos(4 * np.pi * day_of_year/max_val)
    cos3 = np.cos(6 * np.pi * day_of_year/max_val) 
    cos4 = np.cos(8 * np.pi * day_of_year/max_val)
    
    return np.array([sin1, sin2, sin3, sin4, cos1, cos2, cos3, cos4])


#Chưa hiểu
def create_weights(w,data):
    time_len = data.shape[0] #Tất cả mốc tgian
    params = Hyperparams(hybridize=True) #Lấy params
    stride_size = 48 #Dự đoán 48 điểm tiếp theo 
    window_size = params.pred_days*4*stride_size
    input_size = window_size-stride_size
    total_windows = (time_len-input_size) 
    
    weight_indic = np.zeros((total_windows, 1), dtype='float32')   

    count = 0
    for i in range(input_size,total_windows):
        window_start = i - input_size
        window_end = i

        weight_indic[count] = (np.round(w[window_start:window_end, :].sum())).astype(int)
        count += 1
    return weight_indic




##############################################################################################
# SAVE THE DATA
##############################################################################################
def create_data_files(poids):
	covariates, weights = gen_covariates(poids,data_frame.index, num_covariates=96)
	covariates = pd.DataFrame(covariates)
	covariates = covariates.set_index(data_frame.index)

	params = Hyperparams(hybridize=True)
	empty_weights =  np.zeros(shape=(params.pred_days*3*48,1)) 
	weight_indic = pd.DataFrame(np.concatenate((empty_weights,create_weights(weights,data_frame)), axis=0))
	weight_indic = weight_indic.set_index(data_frame.index)



	data_frame.to_csv('Count_data.csv') #OK đã hiểu -> Đã có
	covariates.to_csv('Covariates.csv') #OK đã hiểu -> Biết làm
	weight_indic.to_csv('weight_indic.csv') # ?




