import pandas as pd
import numpy as np
from config import *
from functools import reduce

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score

df = pd.read_csv(path_arquivo_treino)
Xraw = df.drop(columns=['claim_amount'])
yraw = df['claim_amount'].values

def concat_by_id(df):
    dfs=[]
    lista_info=[]
    df_ano_1 = df[df['year']==1]
    if int(max(df['year']))==1:
        df_merged = df
    else:
        for i in range(1,int(max(df['year']))): #add anos 1-max
            df_aux = df[df['year']==i]
            df_aux = df_aux.rename(columns={'claim_amount':'claim_amount_'+str(i),'pol_no_claims_discount':'pol_no_claims_discount_'+str(i)})
            dfs.append(df_aux[['id_policy','claim_amount_'+str(i),'pol_no_claims_discount_'+str(i)]])
            lista_info = lista_info + ['claim_amount_'+str(i),'pol_no_claims_discount_'+str(i)]
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

def carros_prob(df):
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

def preprocessing(Xraw,yraw):
    df_total = Xraw
    df_total['claim_amount'] = yraw #gerando o df original
    carros_maior_prob,carros_menor_prob = carros_prob(df_total) #gerando a lista de carros
    df_total = df_total.drop(df_total[df_total.vh_weight < 50].index) # drop de dados sem sentido
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
    Xraw = df_aux[features+lista_info].copy()
    yraw = ((df_aux['y']>0)).astype(int)
    Xraw.fillna(0,inplace = True)
    
    return Xraw,yraw

def fit_model(X_train,y_train):
    w = {0:1,1:8.5}
    model = RandomForestClassifier(n_estimators=1000,max_depth=5,verbose=1,class_weight=w,random_state=0)
    model.fit(X_train,y_train)
    return model

def predict_claim(model,X_test):
    y_pred = model.predict(X_test)
    return y_pred

def model_metrics(model,y_pred,X_test,y_test):
    score = model.score(X_test, y_test)
    matriz = confusion_matrix(y_test,y_pred,normalize='true')
    
    print(score)
    print(matriz)
    print(list(zip(Xraw.columns,model.feature_importances_)))

if __name__ == "__main__": 
    #pre processamento dos dados
    Xraw_1,yraw_1 = preprocessing(Xraw,yraw)

    #divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(Xraw_1, yraw_1, test_size=0.33, random_state=0)

    #fit e predição do modelo
    model = fit_model(X_train,y_train)
    y_pred_1 = predict_claim(model,X_test)

    #função de métricas do modelo
    model_metrics(model,y_pred_1,X_test,y_test)