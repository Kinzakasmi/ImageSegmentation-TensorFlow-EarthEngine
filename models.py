import tensorflow as tf
class ModelTrainingAndEvaluation():
    def __init__(self,model_name,train,eval,finetune):
        '''Arguments :
            train       : dictionary of training dataset
            model_name  : the name of the model to be saved as, it must contain 'knn'. Example 'knn_sentinel8_sentinel1_20bands'
        '''
        self.model_name   = model_name
        self.train        = train
        self.eval         = eval
        self.finetune     = finetune
        self.num_features = train['features'].shape[-1]
        self.model        = None
        self.reducer      = None
        

    def knn(self,n_neighbors=50,weights='distance',metric='euclidean'):
        '''Function that trains a kNN classifier and saves it
        Arguments :
            n_neighbors : integer, default to 50, irrelevant if finetune=True
            weights     : string, default to 'distance', irrelevant if finetune=True
            metric      : string, default to 'euclidean', irrelevant if finetune=True
        Returns :
            trained kNN model
        '''
        from sklearn.neighbors import KNeighborsClassifier 
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import balanced_accuracy_score,f1_score
        from joblib import dump

        x_train = self.train['features']
        y_train = self.train['labels']
        scorer  = make_scorer(f1_score,average='weighted')
        
        if self.finetune :
            parameters = {'n_neighbors':[5, 20,50,100], #K SMOOTHES THE DECISION SURFACE => UNDERFITS
                        'metric': ['euclidean','manhattan']
                        }
            clf = RandomizedSearchCV(
                KNeighborsClassifier(weights='distance'),parameters,n_iter=2,cv=4,scoring=scorer,n_jobs=5,verbose=10)
            clf.fit(x_train,y_train)

            print('best params : ',clf.best_params_)
            print('best validation score : ',clf.best_score_)
            model = clf.best_estimator_
            del clf
        else :
            model = KNeighborsClassifier(n_neighbors=n_neighbors,weights=weights,metric=metric)
        
        model.fit(x_train,y_train)
        #print('train score : ',model.score(x_train, y_train))

        # save model
        dump(model, 'models/'+self.model_name+'.joblib') 
        self.model = model

    def svm(self,C=25,kernel='rbf',gamma=1e-3):
        '''Function that trains a SVM classifier and saves it
        Arguments :
            C          : float, default to 25, irrelevant if finetune=True
            kernel     : string, default to 'rbf', irrelevant if finetune=True
            gamma      : float, default to 1e-3, irrelevant if finetune=True
        Returns :
            trained svm model
        '''
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import balanced_accuracy_score,f1_score
        from joblib import dump

        x_train = self.train['features']
        y_train = self.train['labels']
        scorer = make_scorer(f1_score,average='weighted')
        
        if self.finetune :
            if 'linearsvm' in self.model_name :
                parameters = {'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                clf = RandomizedSearchCV(
                   LinearSVC(multi_class='ovr',dual=False,loss='squared_hinge'),
                    parameters,n_iter=8,cv=3,scoring=scorer,n_jobs=-1,verbose=10)
            else :
                parameters = [
                            {'kernel': ['rbf','sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                                'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                            {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}]
                clf = RandomizedSearchCV(SVC(),parameters,n_iter=6,cv=2,scoring=scorer,n_jobs=5,verbose=50)
            
            clf.fit(x_train,y_train)
            print('best params : ',clf.best_params_)
            print('best validation score : ',clf.best_score_)
            model = clf.best_estimator_
            del clf
        else :
            if 'linearsvm' in self.model_name :
                model = LinearSVC(C=C,dual=False,verbose=10)
            else :
                model = SVC(C=C,kernel=kernel,gamma=gamma,cache_size=500,verbose=10)

        model.fit(x_train,y_train)
        #print('train score : ',model.score(x_train, y_train))

        # save model
        dump(model, 'models/'+self.model_name+'.joblib')
        self.model = model

    def rf(self,n_estimators=50, max_features='auto', max_depth=16, criterion='gini') :
        '''Function that trains a random forest classifier and saves it
        Arguments :
            n_estimators : integer, default to 50 
            max_features : string, default to 'auto' 
            max_depth : integer, default to 16 
            criterion : string, default to 'gini'
        Returns :
            trained random forest model
        '''
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import balanced_accuracy_score,f1_score
        from joblib import dump

        x_train = self.train['features']
        y_train = self.train['labels']
        scorer = make_scorer(f1_score,average='weighted')
        
        if self.finetune :
            parameters = {'n_estimators': [20, 50,100, 150],
                        'max_features': ['auto'],
                        'max_depth' : [10,16,18],
                        'criterion' :['gini']
                        }

            model = RandomForestClassifier()
            clf = RandomizedSearchCV(model,parameters,n_iter=3,cv=3,scoring=scorer,n_jobs=7,verbose=50)
            clf.fit(x_train,y_train)

            print('best params : ',clf.best_params_)
            print('best validation score : ',clf.best_score_)
            model = clf.best_estimator_
            del clf
        else :
            model = RandomForestClassifier(n_estimators=50, max_features='auto', max_depth=16, criterion='gini')
        
        model.fit(x_train,y_train)
        #print('train score : ',model.score(x_train, y_train))

        # save model
        dump(model, 'models/'+self.model_name+'.joblib')
        self.model = model

    def pca(self,num_classes):
        import os 
        from sklearn.decomposition import PCA
        from joblib import dump,load
        from copy import deepcopy
        from dataset_loader import undersample

        if os.path.isfile('models/'+self.model_name+'.sav'):
            reducer = load('models/'+self.model_name+'.sav')
        else :
            print('Training PCA...')
            reducer = PCA(n_components='mle',copy=True,svd_solver='auto')
            train_pca = deepcopy(self.train)
            train_pca['features'], train_pca['labels'] = undersample(train_pca['features'], train_pca['labels'],num_classes,samples_per_class=270000) #too long if we don't reduce the number of samples
            reducer.fit(train_pca['features'])
            dump(reducer,'models/'+self.model_name+'.sav')
            del train_pca

        self.reducer = reducer
        self.num_features = reducer.n_components_

        print('Reducing dimension...')
        if self.train['features'].shape[-1] != self.num_features :
            self.train['features'] = reducer.transform(self.train['features'])

    def umap(self,num_classes):
        import os
        import umap
        from joblib import dump,load
        from copy import deepcopy
        from dataset_loader import undersample
        
        if os.path.isfile('models/'+self.model_name+'.sav'):
            reducer = load('models/'+self.model_name+'.sav')
        else :
            print('Training UMAP...')
            reducer = umap.UMAP(n_components=5,n_neighbors=15,target_weight=0.7,n_epochs=250,low_memory=True,verbose=True)
            train_pca = deepcopy(self.train)
            train_pca['features'], train_pca['labels'] = undersample(train_pca['features'], train_pca['labels'],num_classes,samples_per_class=270000)
            reducer.fit(train_pca['features'],y=train_pca['labels'])
            dump(reducer,'models/'+self.model_name+'.sav')
            del train_pca
        
        self.num_features = self.train['features'].shape[-1]
        self.reducer = reducer
        
        print('Reducing dimension...')
        self.train['features'] = reducer.transform(self.train['features'])
    
    def plot_feature_importance(self): 
        import matplotlib.pyplot as plt
        import numpy as np

        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(self.num_features):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        # Plot the impurity-based feature importances of the forest
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(self.num_features), importances[indices],
                color="r", yerr=std[indices], align="center")
        plt.xticks(range(self.num_features), indices)
        plt.xlim([-1, self.num_features])
        plt.savefig('results/'+self.model_name+'_feature_importance')
        plt.show()

    def get_labels(self) :
        y_true = self.eval['labels']
        y_pred = self.model.predict(self.eval['features'])
        return y_true, y_pred
    
    def get_performance(self,y_true,y_pred,label_names,target_names):
        from sklearn.metrics import classification_report
        import pandas as pd
        results = classification_report(y_true,y_pred,labels=label_names,target_names=target_names,output_dict=True)
        df = pd.DataFrame.from_dict(results).transpose()
        df['precision'] *= 100
        df['recall'] *= 100
        df['f1-score'] *= 100
        df = df.round(2)
        print(df)
    
    def visualize_confusion_matrix(self,y_true,y_pred_argmax,name):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        cm = tf.math.confusion_matrix(y_true,y_pred_argmax).numpy()
        con_mat_df = pd.DataFrame(cm)
        sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('results/'+name+'_confusion_matrix.png')
        plt.show()
        return con_mat_df

    def eval_model(self,label_names,target_names):
        print('Evaluating the model...')
        if self.reducer :
            self.eval['features'] = self.reducer.transform(self.eval['features'])

        x,y = self.eval['features'],self.eval['labels']
        y_pred = self.model.predict(x)
        self.get_performance(y,y_pred,label_names,target_names)
        self.visualize_confusion_matrix(y,y_pred,self.model_name)