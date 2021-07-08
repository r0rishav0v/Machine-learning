import numpy as np
#from numba import jit, cuda
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearRegression() :
      
    def __init__( self, learning_rate) :
        self.learning_rate = learning_rate
          
    #@cuda.jit         
    def fit( self, X, Y ) :
        self.m, self.n = X.shape
        self.W = np.zeros( self.n )
        self.b = 0
        self.X = X
        self.Y = Y
        count = 0
        
        Y_pred = self.predict( self.X )
        Loss = sum(np.sqrt((Y - Y_pred)**2))/len(Y)
        print("First Loss")
        print(Loss)
        self.update_weights()
        Y_pred = self.predict( self.X )
        New_Loss = sum(np.sqrt((Y - Y_pred)**2))/len(Y)
        print("Second Loss")
        print(New_Loss)
        while(New_Loss < Loss):
            Loss = New_Loss
            self.update_weights()
            Y_pred = self.predict( self.X )
            New_Loss = sum(np.sqrt((Y - Y_pred)**2))/len(Y)
            print("Loss")
            print(New_Loss)
            count = count + 1
            print(count)
        return self
      
    # Helper function to update weights in gradient descent
      
    def update_weights( self ) :
        Y_pred = self.predict( self.X )
        # calculate gradients  
        dW = - ( 2 * ( self.X.T ).dot( self.Y - Y_pred )  ) / self.m
        db = - 2 * np.sum( self.Y - Y_pred ) / self.m 
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
      
    # Hypothetical function  h( x ) 
      
    def predict( self, X ) :
        return X.dot( self.W ) + self.b
		
		

def main() :
    
    df = pd.read_csv('W:/Kaggle/References-master/References-master/Fake_data.csv')
    X = df.iloc[:,:-1].values
    Y = df.iloc[:,5].values
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 1/3, random_state = 0 )
    # Model training
    model = LinearRegression(learning_rate = 0.025 )
    model.fit( X_train, Y_train )
    # Prediction on test set
    Y_pred = model.predict( X_test )
    print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
    print( "Real values      ", Y_test[:3] )
    print( "Trained W        ", model.W) 
    print( "Trained b        ", round( model.b, 2 ) )
    # Visualization on test set 
    print(Y_test)
    print(Y_pred)
    #print(model.Y_pred)
    
    print(df)
    #plt.scatter( X_test, Y_test, color = 'blue' )
    #plt.plot( X_test, Y_pred, color = 'orange' )
    #plt.title( 'Salary vs Experience' )
    #plt.xlabel( 'Years of Experience' )
    #plt.ylabel( 'Salary' )
    #plt.show()
     
if __name__ == "__main__" : 
    main()