from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor



class ManualModels:

    def decision_tree(self, X_train, y_train):
    
        # initialise model
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)# this is what allows the model to be trained on the train data 

        return model #returns the trained model
    
    def naive_bayes(self, X_train, y_train):

        # initialise model
        model = GaussianNB()
        model.fit(X_train, y_train)# this is what allows the model to be trained on the train data 


        return model #returns trained model
    
    def mlp(self, x_train, y_train):
        
        model = MLPRegressor(
            activation= "relu", #this is to help the model learn non-linear relationships
            max_iter = 300, #max training rounds so that the model can learn 
            random_state=42,  #fixes randomness so its reproducible
            early_stopping=True,#stops training if results are improving
            validation_fraction= 0.1, #sets aside 10% of the data to check performance
            verbose=False #hides training logs (just gives a cleaner output)
        )
        
            

        model.fit(x_train, y_train)

        return model






    

