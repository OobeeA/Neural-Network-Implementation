import torch
import pickle
import numpy as np
import pandas as pd

# Python libraries we've imported
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing

from sklearn.metrics import r2_score

class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        self.transformer = None
        self.scalar = None
        self.categories = [('ocean_proximity',['INLAND','<1H OCEAN','NEAR BAY', 'NEAR OCEAN', 'ISLAND'])]
        self.model = None

        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or 
                testing the model.

        Returns:
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
              size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
            - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
              size (batch_size, 1).
            
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None


        # If it's a training set, we can use this to create our one hot encoding for all our datasets that we'll be using it to train

        #data = pd.concat([X, Y], axis=1)

        ## Shuffle the dataset
        #shuffled_data = shuffle(data)

        # Split the shuffled dataset back into X and Y
        #X_shuffled = shuffled_data.drop(columns=[Y.columns[0]])
        #Y_shuffled = shuffled_data[Y.columns[0]]


        if training:
            ohe_columns = [a[0] for a in self.categories]
            ohe_categories = [a[1] for a in self.categories]
            
            # we will drop any rows that have null values. The reason why we have chose to drop the rows is that there are 
            # only 10 such rows.
            transformed_x = x.dropna(
                axis=0,
                how='any',
                subset=None,
                inplace=False
            )
            if isinstance(y, pd.DataFrame): transformed_y = y.loc[transformed_x.index]

            # we will one hot encode the feature ocean_proximity
            enc = OneHotEncoder(sparse_output=False, categories=ohe_categories)
            self.transformer = make_column_transformer((enc, ohe_columns), remainder='passthrough')
            transformed_x = self.transformer.fit_transform(transformed_x)

            # we will now standardise every feature ensuring they have a mean of 0 and a standard deviation of 1
            self.scaler = preprocessing.StandardScaler().fit(transformed_x)
            transformed_x = self.scaler.transform(transformed_x)

            # Transorming the numpy array into scalar 
            transformed_x = torch.from_numpy(transformed_x)
            transformed_x = transformed_x.to(torch.float32)

            # convert y into a torch tensor
            if isinstance(y, pd.DataFrame):
                transformed_y = torch.from_numpy(transformed_y.values)
                transformed_y = transformed_y.to(torch.float32)
                
        else:
            # Drop any null rows
            transformed_y = y
            transformed_x = x.dropna(
                axis=0,
                how='any',
                subset=None,
                inplace=False
            )
            if isinstance(y, pd.DataFrame): transformed_y = y.loc[transformed_x.index]

            # One hot encode any features that may need to be encoded
            transformed_x = self.transformer.transform(transformed_x)

            # scale the features using scaling parameters from the training set
            transformed_x = self.scaler.transform(transformed_x)

            # Transorming the numpy array into scalar 
            transformed_x = torch.from_numpy(transformed_x)
            transformed_x = transformed_x.to(torch.float32)

            if isinstance(y, pd.DataFrame):
                transformed_y = torch.from_numpy(transformed_y.values)
                transformed_y = transformed_y.to(torch.float32)

        return transformed_x, (transformed_y if isinstance(y, pd.DataFrame) else None)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        layers = [13, 1024, 512, 256, 32, 1]
        #layers = [13, 50, 50, 1]
        #layers = [13, 19, 19, 19, 19, 1]
        self.model = torch.nn.Sequential()
        for i in range(len(layers)-2):
            self.model.append(torch.nn.Linear(layers[i], layers[i+1]))
            self.model.append(torch.nn.ReLU())
        self.model.append(torch.nn.Linear(layers[-2], layers[-1]))

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        #optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


        n_epochs = 500
        batch_size = 20000

        isSet = False
        isSet1 = False
        for epoch in range(n_epochs):
            for i in range(0, len(X), batch_size):
            #for i in range(0, 1, batch_size):
                Xbatch = X[i:i+batch_size]
                y_pred = self.model(Xbatch)
                ybatch = Y[i:i+batch_size]
                #print(y_pred, ybatch)
                loss = loss_fn(y_pred, ybatch)
                print(epoch)

                if loss ** 0.5 < 50000 and not isSet:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
                    isSet = True

                if loss ** 0.5 < 40000 and not isSet1:
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
                    isSet1 = True

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #if loss ** 0.5 < 50000:
                #    break
            #if loss ** 0.5 < 50000:
            #    break
            #with torch.no_grad():
            #     y_pred = self.model(X)
            #     accuracy = (y_pred.round() == Y).float().mean()
            #     print(f"Accuracy {accuracy}")
        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        return self.model(X)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget


        loss_fn_mse = torch.nn.MSELoss()
        loss_fn_l1 = torch.nn.L1Loss(reduction = "mean")

        y_pred = self.model(X)
        rmse = loss_fn_mse(y_pred, Y) ** 0.5
        l1_error = loss_fn_l1(y_pred, Y)
        #r2 = r2_score(Y.detach().numpy(), y_pred.detach().numpy())
        print("The rmse", rmse)
        print("The l1 error", l1_error)
        #print("The r2", r2)

        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    output_label = "median_house_value"
    #example_main()
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)
    regressor.fit(x_train, y_train)
    regressor.score(x_train, y_train)


