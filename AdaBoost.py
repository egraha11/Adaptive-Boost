import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class AdaBoost:

    def __init__(self, test_names):

        self.alpha_weights = []
        self.stump_forest = []
        self.iterations = 100
        self.test_names = test_names
        self.final_preds = None




    def calculate_alpha(self, preds, y, weights):

        #calculate the sum of the total error weights
        errors = sum(weights * np.not_equal(preds, y).astype(int))

        #return alpah value that will determine  the ammount of say this particular stump will have 
        return (1/2 * np.log((1 - errors) / errors))



    def calculate_weights(self, preds, y, weights, alpha):

        #save indices where the current stump predicted incorrectly and correctly to update weights for the next stump
        error_indices = np.where(preds != y)
        accurate_indices = np.where(preds == y)

        #calculate new weights for nodes that need to be correctly predicted by the next stump
        self.weights[error_indices] = self.weights[error_indices] * np.exp(alpha)
        self.weights[accurate_indices] = self.weights[accurate_indices] * np.exp(-(alpha))

        #normalize the weights so each models alpha can be calculated independently 
        self.weights = self.weights/sum(self.weights)

        


    def fit(self, x, y):

        self.x = x
        self.y = y

        #initialize starting weights
        self.weights = np.full(len(x), 1/len(x))

        #fit x number of weak learners
        for i in range(self.iterations):

            stump = DecisionTreeClassifier(max_depth=1)

            stump.fit(self.x, self.y, sample_weight=self.weights)

            #save stump in the random forest 
            self.stump_forest.append(stump)

            preds = stump.predict(x)

            #use current stumps predictions to calculate new model weight
            new_alpha = self.calculate_alpha(preds, y, self.weights)

            self.alpha_weights.append(new_alpha)

            #use current stums predictions to calculate new node weights
            self.calculate_weights(preds, y, self.weights, new_alpha)

    
    def accuracy(self, actuals):

        actuals = pd.Series(actuals, name="index")
        preds = pd.Series(self.final_preds.iloc[:, -1], name="index")

        print((actuals.reset_index(drop=True) == preds.reset_index(drop=True)).value_counts())



    def predict(self, x):

        #create a new dataframe to store each test nodes predictions 
        predictions = pd.DataFrame(index=range(len(x)), columns=range(len(self.stump_forest)))

        #scores= {key: None for key in np.unique(self.y)}

        for stump in range(len(self.stump_forest)):

            #using all stumps in the forest, predict target values for all test nodes
            predictions.iloc[:, stump] = self.stump_forest[stump].predict(x)


        #substitute categoryical value of 0s with -1s to calculate weighted predictions 
        predictions = predictions.replace(0, -1)

        #get weighted predictions for each prediction for each node
        final_preds = np.dot(predictions, np.array(self.alpha_weights).T)

        #replace weighted predictions with categorical values 
        final_preds[final_preds <= 0] = 0
        final_preds[final_preds > 0] = 1

        self.final_preds = pd.DataFrame(list(zip(self.test_names, final_preds)), columns = ["Name", "Longer Than 5 Years?"])

        print(self.final_preds)


def main():


    df = pd.read_csv("nba_logreg1.csv")

    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.target_5yrs, test_size=.25)

    test_names = x_test.iloc[:, 1]

    names = x_test.iloc[:, 1]

    x_test.drop(x_test.iloc[:, :2], axis=1, inplace=True)
    x_train.drop(x_train.iloc[:, :2], axis=1, inplace=True)

    model = AdaBoost(test_names)

    model.fit(x_train, y_train)

    model.predict(x_test)

    model.accuracy(y_test)


if __name__ == "__main__":
    main()

