from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression



class ManualModels:

    def decision_tree(self, X_train, y_train):
    
    # initialize model
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

        return model
    
    def naive_bayes(self, X_train, y_train):

        # initialize model
        model = GaussianNB()
        model.fit(X_train, y_train)

        return model





    

