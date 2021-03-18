"""In this module, we ask you to define your pricing model, in Python."""

import pickle
import numpy as np
import pandas as pd
from functools import reduce

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# NOTE THAT ANY TENSORFLOW VERSION HAS TO BE LOWER THAN 2.4. So 2.3XXX would work.

# TODO: import your modules here.
# Don't forget to add them to requirements.txt before submitting.



# Feel free to create any number of other functions, constants and classes to use
# in your model (e.g., a preprocessing function).
def concat_by_id(df):
    dfs=[]
    lista_info=[]
    df_ano_1 = df[df['year']==1]
    if int(max(df['year']))==1:
        df_merged = df
    else:
        for i in range(1,int(max(df['year']))): #add anos 1-max
            df_aux = df[df['year']==i]
            df_aux = df_aux.rename(columns={'pol_no_claims_discount':'pol_no_claims_discount_'+str(i)})
            dfs.append(df_aux[['id_policy','pol_no_claims_discount_'+str(i)]])
            lista_info = lista_info + ['pol_no_claims_discount_'+str(i)]
        df_ano_pred = df[df['year']==max(df['year'])] #add ano de predição
        df_ano_pred = df_ano_pred.rename(columns={'claim_amount':'claim_amount_'+str(int(max(df['year'])))})
        dfs.append(df_ano_pred[['id_policy','claim_amount_'+str(int(max(df['year'])))]])
        
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id_policy'],
                                            how='inner'), dfs)
        
        dfs = [df_ano_1[['id_policy', 'pol_coverage',
       'pol_duration', 'pol_sit_duration', 'pol_pay_freq', 'pol_payd',
       'pol_usage', 'drv_sex1', 'drv_age1', 'drv_age_lic1', 'drv_drv2',
       'drv_sex2', 'drv_age2', 'drv_age_lic2', 'vh_make_model', 'vh_age',
       'vh_fuel', 'vh_type', 'vh_speed', 'vh_value', 'vh_weight', 'population',
       'town_surface_area']],df_merged]
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['id_policy'],
                                            how='inner'), dfs)
    return df_merged,lista_info

def carros_prob():
    carros_maior_prob = ['aewtdnpoiopumymt','asmpttrlkodaejic','dgwbxitzfzbegnoc',
                         'dlrodwgixwmoquny','mdxtphkujabwpjeu','oryfrzxilushvigq',
                         'qxtqrwxfvuenelml','sguprofjftozaujc','synvsxhrexuyxpre',
                         'wehkqzwvbeonajcu','wxzfbqtarfurwcfw','yvlkrzgjhwrlyihc']
    carros_menor_prob = ['cgkclpnidlmetsrb','dllcylnkzeegtsgr','dwsasdexwmpsmowl',
                         'epbwnmcyogpybxlm','hpohizpkyzvwunni','hwldevoubgzgbhgs',
                         'jgkpiuuctpywtrlh','ldxjynecsqlswvbq','nhwgapjtnadqqaul',
                         'pjbnwqhnqczouirt','prtnwsypyfnshpqx','shemwbbeliuvnvvm',
                         'wyqgeeclrqbihfpk','zxvcbwcwoqnkxxbs']
    return carros_maior_prob,carros_menor_prob
def preprocessing_pred(Xraw):
    carros_maior_prob,carros_menor_prob = carros_prob()
    df = Xraw
    dfs=[]
    lista_info=[]
    df_ano_1 = df[df['year']==1]
    if int(max(df['year']))==1:
        df_merged_by_id = df.rename(columns={'pol_no_claims_discount':'pol_no_claims_discount_1'})
    else:
        for i in range(1,int(max(df['year']))): #add anos 1-max
            df_aux = df[df['year']==i]
            df_aux = df_aux.rename(columns={'pol_no_claims_discount':'pol_no_claims_discount_'+str(i)})
            dfs.append(df_aux[['id_policy','pol_no_claims_discount_'+str(i)]])
            
        dfs.append(df_ano_1[['id_policy', 'pol_coverage',
       'pol_duration', 'pol_sit_duration', 'pol_pay_freq', 'pol_payd',
       'pol_usage', 'drv_sex1', 'drv_age1', 'drv_age_lic1', 'drv_drv2',
       'drv_sex2', 'drv_age2', 'drv_age_lic2', 'vh_make_model', 'vh_age',
       'vh_fuel', 'vh_type', 'vh_speed', 'vh_value', 'vh_weight', 'population',
       'town_surface_area']])
        
        df_merged_by_id = reduce(lambda  left,right: pd.merge(left,right,on=['id_policy'],
                                            how='inner'), dfs)
        
    df_merged_by_id['pol_coverage_2'] = df_merged_by_id['pol_coverage'].map({'Min':1,'Med1':2,'Med2':3,'Max':4})
    
    df_fuel = pd.get_dummies(df_merged_by_id['vh_fuel'], drop_first = True, prefix = 'fuel')
    df_merged_by_id = pd.concat([df_merged_by_id,df_fuel], axis=1)
    
    df_merged_by_id['grupo_risco_1'] = ((df_merged_by_id['pol_usage']=='Professional')&
                                        (df_merged_by_id['pol_coverage']=='Max')).astype(int)
    
    df_merged_by_id['grupo_risco_2'] = ((df_merged_by_id['pol_usage']=='WorkPrivate')&
                                        (df_merged_by_id['pol_coverage']=='Min')).astype(int)
    
    df_merged_by_id['grupo_risco_3'] = df_merged_by_id['vh_make_model'].isin(carros_maior_prob).astype(int)
    
    df_merged_by_id['grupo_risco_4'] = df_merged_by_id['vh_make_model'].isin(carros_menor_prob).astype(int)
    
    df_merged_by_id['grupo_risco_5'] = ((df_merged_by_id['vh_age']>15) ).astype(int)
    
    df_merged_by_id['grupo_risco_6'] = ((df_merged_by_id['vh_age']>0)&
                                        (df_merged_by_id['vh_age']<6)).astype(int)
    
    df_merged_by_id['vh_value_risk'] = df_merged_by_id['vh_value']/df_merged_by_id['pol_coverage_2']
    
    features = ['id_policy','pol_coverage_2','drv_age_lic1', 'vh_weight',
                'fuel_Gasoline','drv_age1','vh_speed','vh_age',
                'vh_value','population','grupo_risco_1',
                'grupo_risco_2','grupo_risco_3','grupo_risco_4',
                'grupo_risco_5','grupo_risco_6','vh_value_risk']
    
    Xraw = df_merged_by_id[features].copy()
    Xraw.fillna(0,inplace = True)
    
    return Xraw
    

def preprocessing_fit(Xraw,yraw,clas):
    df_total = Xraw
    df_total['claim_amount'] = yraw #gerando o df original
    carros_maior_prob,carros_menor_prob = carros_prob() #gerando a lista de carros
    if clas==0: # se estou no modelo de regressão
        lista_drop_claim_0 = list(df_total[(df_total['year']==int(max(df_total['year'])))&(df_total['claim_amount']>0)]['id_policy'].values)
        df_total = df_total[df_total['id_policy'].isin(lista_drop_claim_0)] #drop claim 0 no ano de predição
    df_merged_by_id,lista_info = concat_by_id(df_total) #df id por linha
    
    df_merged_by_id['pol_coverage_2'] = df_merged_by_id['pol_coverage'].map({'Min':1,'Med1':2,'Med2':3,'Max':4})
    
    df_fuel = pd.get_dummies(df_merged_by_id['vh_fuel'], drop_first = True, prefix = 'fuel')
    df_merged_by_id = pd.concat([df_merged_by_id,df_fuel], axis=1)
    
    df_merged_by_id['grupo_risco_1'] = ((df_merged_by_id['pol_usage']=='Professional')&
                                        (df_merged_by_id['pol_coverage']=='Max')).astype(int)
    
    df_merged_by_id['grupo_risco_2'] = ((df_merged_by_id['pol_usage']=='WorkPrivate')&
                                        (df_merged_by_id['pol_coverage']=='Min')).astype(int)
    
    df_merged_by_id['grupo_risco_3'] = df_merged_by_id['vh_make_model'].isin(carros_maior_prob).astype(int)
    
    df_merged_by_id['grupo_risco_4'] = df_merged_by_id['vh_make_model'].isin(carros_menor_prob).astype(int)
    
    df_merged_by_id['grupo_risco_5'] = ((df_merged_by_id['vh_age']>15) ).astype(int)
    
    df_merged_by_id['grupo_risco_6'] = ((df_merged_by_id['vh_age']>0)&
                                        (df_merged_by_id['vh_age']<6)).astype(int)
    
    df_merged_by_id['vh_value_risk'] = df_merged_by_id['vh_value']/df_merged_by_id['pol_coverage_2']
    
    features = ['pol_coverage_2','drv_age_lic1', 'vh_weight',
                'fuel_Gasoline','drv_age1','vh_speed','vh_age',
                'vh_value','population','grupo_risco_1',
                'grupo_risco_2','grupo_risco_3','grupo_risco_4',
                'grupo_risco_5','grupo_risco_6','vh_value_risk']
    
    df_aux = df_merged_by_id.rename(columns = {'claim_amount_'+str(int(max(df_total['year']))):'y'})
    Xraw = df_aux[features].copy()
    if clas==0:
        yraw = df_aux['y']
    else:
        yraw = (df_aux['y']>0).astype(int)
    Xraw.fillna(0,inplace = True)
    
    return Xraw,yraw



def fit_model(X_raw, y_raw):
    X_class,y_class = preprocessing_fit(X_raw,y_raw,1)
    X_reg,y_reg = preprocessing_fit(X_raw,y_raw,0)
    
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.33, random_state=0)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.33, random_state=0)
    
    w = {0:1,1:8.5}
    model_class = RandomForestClassifier(n_estimators=1000,max_depth=5,verbose=1,class_weight=w,random_state=0)
    model_class.fit(X_train_class,y_train_class)
    
    model_reg = LinearRegression().fit(X_train_reg, y_train_reg)
    
    model = [model_class,model_reg]

    return model 



def predict_expected_claim(model, X_raw):

    preds=[]
    for ano in range(1,int(max(X_raw['year'])+1)):
        X_ano = X_raw[X_raw['year']<=ano].copy()
        X_ano = preprocessing_pred(X_ano)

        df_ano = pd.DataFrame()
        df_ano['id_policy'] = X_ano['id_policy'].copy()
        df_ano['year'] = ano

        model_class = model[0]
        model_reg = model[1]
        
        X_ano.drop('id_policy',axis=1,inplace=True)

        y_pred_prob = model_class.predict_proba(X_ano)[:,1]
        y_pred_claim = model_reg.predict(X_ano)
        y_pred = y_pred_prob*y_pred_claim

        y_pred = np.where(y_pred < 0,min(y_pred[y_pred>0]), y_pred)

        df_ano['claim'] = y_pred
        preds.append(df_ano.copy())

    df_pred = pd.concat(preds,axis=0)
    
    df_pred = pd.merge(
        X_raw,
        df_pred,
        on=['year','id_policy'],
        how='left'
    )
    
    return df_pred['claim'].values



def predict_premium(model, X_raw):
	return predict_expected_claim(model, X_raw) * 1.7 



def save_model(model_list):
    with open('trained_model.pickle', 'wb') as target:
        for trained_model in model_list:
            pickle.dump(trained_model,target)

            
def load_model():
    trained_model = []
    with open('trained_model.pickle', 'rb') as target:
        while True:
            try:
                trained_model.append(pickle.load(target))
            except EOFError:
                    break
    return trained_model
