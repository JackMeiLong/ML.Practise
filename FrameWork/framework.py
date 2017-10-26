
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
import datetime
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

'''
FrameWork_v1.py
1:preprocess(objet->onehot,float->scaler,add symbol feature)
2:feature_selection_v1(VarianceThreshold)
3:feature_selection_v2(Chi2,F-test,Mutual Info)
4:dimension_reduce(PCA,LDA)
5:feature_transform(GBDT+LR) coming soon

'''

class POS:
     
    def __init__(self):
        self.data_path='/mnt/data/meilong/simulate/'
        self.data=[]
        self.pos_label=[]
        self.label=[]
        self.train=[]
        self.raw_train=[]
        self.train_label=[]
        self.raw_train_label=[]
        self.test=[]
        self.raw_test=[]
        self.test_label=[]
        
    def load(self):
        print 'Load...'
        print 'Start: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print 'End: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pass
      
    def generate_label(self):
        print 'Generate Label...'
        print 'Start: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        
        print 'End: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        pass
    
    def split(self,data,label):
        
        print 'Split Data...'
        print 'Start: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        
        print 'End: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        pass
     
    def preprocess(self):
        print 'Preprocess...'
        print 'Start: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        data = self.data.copy()
        label = self.label.copy()
        
        m = data.shape[0]
        print data['MarriageStatus'].dtype
        
        #fillna
        for i in data.columns:
            if i!='AppId' and i!='InstallmentStartedOn':
                if data[i].hasnans:
                    t0=pd.DataFrame(np.ones((data.shape[0],1),dtype=np.int),columns=[i+'_Ex'],index=data.index)
                    ind0=data[data[i].isnull()].index
                    t0.ix[ind0]=0
                    data[i+'_Ex']=t0
                    
                    if data[i].dtype==np.object:
                        if data[i].value_counts().sort_values().shape[0]>0:
                            data[i].fillna(data[i].value_counts().sort_values().index[-1],inplace=True,downcast='infer')
                        else:
                            data[i].fillna('0',inplace=True,downcast='infer')
                    else:
                        if np.isnan(data[i].mean())==False:
                            data[i].fillna(data[i].mean(),inplace=True,downcast='infer')
                        else:
                            data[i].fillna(0,inplace=True,downcast='infer')
                            
        train,train_label,test,test_label=self.split(data,label)

        self.raw_train=train.copy()
        self.raw_train_label=train_label.copy()
        self.raw_test=test.copy()
        self.raw_test_label=test_label.copy()
        
        #delete AppId and InstallmentStartedOn
        data.drop(['AppId','InstallmentStartedOn'],axis=1,inplace=True)
        train.drop(['AppId','InstallmentStartedOn'],axis=1,inplace=True)
        test.drop(['AppId','InstallmentStartedOn'],axis=1,inplace=True)
        
        data.reset_index(inplace=True,drop=True)
        train.reset_index(inplace=True,drop=True)
        test.reset_index(inplace=True,drop=True)
        
        #preprocess 
        enc0=LabelEncoder()
        enc1 = OneHotEncoder()
        scaler = MinMaxScaler()
        
        for i in train.columns:
            if train[i].dtype==np.object:
                t0=enc0.fit_transform(train[i].values.reshape(-1,1))
                t1=enc1.fit_transform(t0.reshape(-1,1)).toarray()
                tf=pd.DataFrame(t1,index=train.index)
                tf.rename(columns=lambda x: i+'_'+str(x)+'_E', inplace=True)
                train.drop(i,inplace=True,axis=1)
                train=train.join(tf,how='inner')

                clas = enc0.classes_
                if test[i][~test[i].isin(clas)].size != 0:
                    ind = test[i][~test[i].isin(clas)].index
                    test[i].iloc[ind] = clas[0]
                    
                t0=enc0.transform(test[i].values.reshape(-1,1))
                t1=enc1.transform(t0.reshape(-1,1)).toarray()
                tf=pd.DataFrame(t1,index=test.index)
                tf.rename(columns=lambda x: i+'_'+str(x)+'_E', inplace=True)
                test.drop(i,inplace=True,axis=1)
                test=test.join(tf,how='inner')              
            else:
                tt0=train[i].values.reshape(-1,1)
                tt0_s=scaler.fit_transform(tt0)
                train[i+'_S']=tt0_s
                train.drop(i,inplace=True,axis=1)               
               
                tt2=test[i].values.reshape(-1,1)
                tt2_s=scaler.transform(tt2)      
                test[i+'_S']=tt2_s
                test.drop(i,inplace=True,axis=1)
        
        #feature selection
        sel = VarianceThreshold(threshold=0.0002)
        train_new=sel.fit_transform(train)
        sup=sel.get_support()
        features=train.columns.tolist()
        for i in xrange(train.shape[1]):
            if sup[i]==False:
                features.remove(train.columns[i])
        
        train=pd.DataFrame(train_new,columns=features)
        
        test_new=sel.transform(test)
        test=pd.DataFrame(test_new,columns=features)
        
        self.train=train.copy()
        self.train_label=train_label.copy()
        self.test=test.copy()
        self.test_label=test_label.copy()
        
        print 'End: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        return train,train_label,test,test_label
    
    def feature_selection(self,mode='F'):
        
        print 'Feature Selection...'
        print 'Start:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        X=self.train.copy()
        y=self.train_label['label'].values.copy()
        
        test=self.test.copy()
        
        if mode.upper()=='M':
            mi=mutual_info_classif(train.values,train_label['label'].values)
        elif mode.upper()=='F':
            F,pval=f_classif(train.values,train_label['label'].values)
        elif mode.upper()=='C':
            chi,pval=chi2(train.values,train_label['label'].values)
        
        features=self.train.columns.copy()
        
        fs_features=features.copy().tolist()
        
        if mode.upper()=='M':
            fs_V=mi.copy().tolist()
        elif mode.upper()=='F':
            fs_V=F.copy().tolist()
        elif mode.upper()=='C':
            fs_V=chi.copy().tolist()
        
        if mode.upper()=='M':
            selector=SelectPercentile(mutual_info_classif,percentile=80)
        elif mode.upper()=='F':
            selector=SelectPercentile(f_classif,percentile=80)
        elif mode.upper()=='C':
            selector=SelectPercentile(chi2,percentile=80)
            
        X_new=selector.fit_transform(X,y)
        
        selected=selector.get_support()
        
        for i in xrange(len(features)):
            if selected[i]==False:
                t=features[i]
                fs_features.remove(t)
                
        fs_V=np.array(fs_V)
        fs_features=np.array(fs_features)
        
        self.train=pd.DataFrame(X_new,columns=fs_features.tolist())
        self.test=test[fs_features]
        
        self.fs_features=fs_features
        
        feas=pd.DataFrame()
        feas['feature']=fs_features
        
        print 'End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return X_new,feas
        
    def dimension_reduce(self,mode='L'):
        
        print 'Reduce Dimensions...'
        print 'Start:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        raw_train=self.train.copy()
        train=self.train.copy()
        train_label=self.train_label['label'].values.copy()
        train_label=train_label.reshape((train_label.shape[0]))
            
        test=self.test.copy()
        test_label=self.test_label['label'].values.copy()
        test_label=test_label.reshape((test_label.shape[0]))
        
        flist=train.columns
        
        if mode.upper()=='L':
            lda=LinearDiscriminantAnalysis()
            X_new=lda.fit_transform(train.values,train_label)
            self.train=pd.DataFrame(X_new,columns=['DR'])
            self.test=pd.DataFrame(lda.transform(test[flist].values),columns=['DR'])
            
            tt=lda.coef_[0]
            ind=np.argsort(tt)
            features=raw_train.columns[ind[-100:]]
            feas=pd.DataFrame()
            feas['feature']=features
            feas['values']=tt[ind[-100:]]
            return feas
            
        elif mode.upper()=='P':
            pca = PCA(n_components=100)
            X_new=pca.fit_transform(train.values,train_label)
            self.train=pd.DataFrame(X_new)
            self.test=pd.DataFrame(pca.transform(test[flist].values))
            
        print 'End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def feature_transform(self,):
        
        pass
    
    def model_init(self,model='lr'):
        print 'Model Init...'
        print 'Start: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')        
        if model.upper() == 'LR':
            classifier = LogisticRegression()
        elif model.upper() == 'RFC':
            classifier = RandomForestClassifier(n_jobs=3)
        elif model.upper() == 'ABC':
            classifier = AdaBoostClassifier()
        elif model.upper() == 'GBDT':
            classifier = GradientBoostingClassifier()
        elif model.upper() == 'MLP':
            classifier = MLPClassifier()
        elif model.upper()=='XGB':
            classifier=XGBClassifier(nthread=3)

        print 'End: '+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return classifier
    
    def model_select_param(self,model='LR'):

        print 'Model Select Parameters'
        print 'Start:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        clf = self.model_init(model)
        
        if model.upper() == 'RFC':
            param_grid = {'n_estimators': [100, 200, 500], 'criterion': ['entropy', 'gini'],'max_depth': [10, 50, 100],
                         'max_features':[30],'class_weight': ['balanced']}
        elif model.upper() == 'LR':
            param_grid = {'penalty': ['l1', 'l2'],'C': [0.3,0.4,0.5],'class_weight': ['balanced']}
        elif model.upper() == 'ABC':
            param_grid = {'n_estimators': [100,200,500], 'learning_rate': [0.08,0.1,0.3]}
        elif model.upper() == 'GBDT':
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.6], 'max_depth': [10, 50]}
        elif model.upper() == 'MLP':
            param_grid = {'activation': ['logistic','tanh','relu'],'learning_rate': ['constant', 'invscaling', 'adaptive'],
                          'max_iter': [100, 200, 800],'hidden_layer_sizes':[(100,)]}
        elif model.upper()=='XGB':
            param_grid={'n_estimators':[100,200],'gamma':[0.1,0.6],'max_depth':[10,50]}
        
        gs = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=5)
        
        data = self.train.values.copy() 
        label = self.train_label['label'].values.copy()

        label = label.reshape(label.shape[0])       
        gs.fit(data, label)
        
        print 'Model: {0} '.format(model)
           
        print gs.grid_scores_
        print gs.best_score_
        print gs.best_params_
        
        print 'End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
        auc_train=None
        auc_test=None
        
        auc_train=self.model_cross_validation(model,gs.best_params_)
        auc_test=self.model_test(model,gs.best_params_)
        
        return auc_train,auc_test
    
    def model_cross_validation(self, model, best_params):

        print 'Model Cross Validation'
        print 'Start:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        clf = self.model_init(model)

        clf.set_params(**best_params)
        
        train_data = self.train.values.copy() 
        train_label = self.train_label['label'].values.copy()
        
        train_label = train_label.reshape(train_label.shape[0])

        scores = cross_val_score(clf, train_data, train_label, cv=5, scoring='roc_auc', n_jobs=3)
        
        print clf
        print scores
        print np.mean(scores)
        
        print 'Model: {0} ; Train: {1}'.format(model,np.mean(scores))
        
        print 'End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return np.mean(scores)
    
    def model_test(self,model,best_params):

        print 'Model Test'
        print 'Start:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        clf = self.model_init(model)                
        clf.set_params(**best_params)
        
        train_data = self.train.values.copy() 
        train_label = self.train_label['label'].values.copy()
        
        clf.fit(train_data, train_label)
        
        if model.upper()=='LR':
            coef=clf.coef_.reshape(clf.coef_.shape[1])
            ind=coef.argsort()
            att=self.train.columns[ind[-30:]].tolist()
            print att
        elif model.upper()=='RFC':
            imp=clf.feature_importances_
            print imp
            ind=imp.argsort()
            att=self.train.columns[ind[-30:]].tolist()
            print att
        elif model.upper()=='XGB':
            imp=clf.feature_importances_
            print imp
            ind=imp.argsort()
            att=self.train.columns[ind[-30:]].tolist()
            print att           
            
        test_data = self.test.values.copy()
        test_label = self.test_label['label'].values.copy()
        test_label = test_label.reshape(test_label.shape[0])
            
        res_proba=clf.predict_proba(test_data)              
        res_auc=roc_auc_score(test_label,res_proba[:,1])
        
        print 'Model: {0} ; Test: {1}'.format(model,res_auc)
                
        print 'End:' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return res_auc

if __name__=='__main__':
    pos=POS()
    data,pos_label=pos.load()
    all_data,all_label=pos.generate_label()
    train,train_label,test,test_label=pos.preprocess()
    print train.shape
    print test.shape
    
    auc_train,auc_test=pos.model_select_param(model='rfc')
