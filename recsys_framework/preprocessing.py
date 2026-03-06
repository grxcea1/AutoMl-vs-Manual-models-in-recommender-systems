import pandas as pd 
from sklearn.model_selection import train_test_split


class preprocessing:


    # constructor 

    def __init__(self):
        self.ratings = None
        self.users = None
        self.items = None
        self.data = None
        self.X = None
        self.y = None 

    
    def load_ratings(self):
        self.ratings = pd.read_csv("data/u.data.csv", sep="\t",
                      names=["user_id","item_id","rating","timestamp"])
        


    
    def load_users(self):
        self.users = pd.read_csv("data/u.user.csv", sep="|",
                        names=["user_id","age","gender","occupation","zip"])
        


    def load_items(self):
        self.items = pd.read_csv("data/u.item.csv", sep="|", encoding="latin-1",
                        names=["item_id","title","release_date","video_release_date","IMDb_URL",
                               "unknown","Action","Adventure","Animation","Children's","Comedy",
                               "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
                               "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"])
       

    def normalize_data(self):

        # merge datasets into one dataset

        self.data = self.ratings.merge(self.users, on = "user_id")
        self.data = self.data.merge(self.items, on="item_id")

        # Take off irrelevant features/columns from the dataset
        self.data = self.data.drop(columns=["timestamp", "zip", "title", "release_date", "video_release_date", "IMDb_URL"])

        self.data = self.data.dropna() #this handles missing data 
        self.data = self.data.drop_duplicates(subset=["user_id", "item_id"]) #this will handle my duplicates


        # convert genders 'M' and 'F' to (0,1) so it can be used by ML models
        self.data["gender"] = self.data["gender"].map({"M":0, "F":1}) 

        self.data = pd.get_dummies(self.data, columns=["occupation"])#this converts word catergories to binary for the models to use

        

    def data_split(self):

        # predictors
         self.X = self.data.drop(columns=["rating"]) 
         self.y = self.data["rating"] #target


    def load_and_preprocess(self):

        # loading all the datasets
        self.load_ratings()
        self.load_users()
        self.load_items()


        # Merge and clean data
        self.normalize_data()

        # splitting data into X and y
        self.data_split()

        X_train, X_test, y_train, y_test = train_test_split(
        self.X, 
        self.y, 
        test_size=0.2,
        random_state=42)


        return X_train, X_test, y_train, y_test



# Run preprocessing pipeline 
test = preprocessing() 


#test, train, split on normalised data
X_train, X_test, y_train, y_test = test.load_and_preprocess() 


#printing the size of train data
print("Training feature shape:", X_train.shape) 


#printing the size of the test data
print("Test feature shape:", X_test.shape) 


#printing first 5 lines of training data 
print("\nSample training data:") 
print(X_train.head()) 


#printing first5 lines for the ratings of the training data 
print("\nSample ratings:") 
print(y_train.head()) 



