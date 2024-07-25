# logistic regression model
class LogisticRegression:
    def __init__(self, epochs=1000, alpha=0.1):
        self.epochs = epochs
        self.alpha = alpha
        self.beta = None
        self.losses = []
        
    # sigmoid function
    def sigmoid(self, x):
        """
        Args:
            x (numpy array): the linear combination of features

        Returns:
            return a probability (0-1)
        """
        return 1 / (1 + np.exp(-x))
    
    # y_hat function
    def y_hat(self, X):
        """
        Args:
            X (numpy array): training data, dimension m x n

        Returns:
            return predicted probabilities, dimension 1 x n
        """
        return sigmoid(np.dot(X, self.beta))
                 
    # cost function per row of data
    def cost_per_row(self, y, yhat):
        if y == 1:
            return np.multiply(y, np.log(yhat)) 
        else:
            return np.multiply(1-y, np.log(1 - yhat))
                 
    # total cost
    def cost(self, Y, Yhat):
        """
        Args:
            Y (numpy array): labels of training data, dimension 1 x m
            Yhat (numpy array): predictied values of training data, dimension 1 x m

        Returns:
            return the cost of beta
        """
        m = len(y)
        df1 = pd.DataFrame(data={'y': Y.ravel(), 'yhat': Yhat.ravel()})
        diff = df1.apply(lambda row: cost_per_row(row['y'], row['yhat']), axis=1).to_numpy().reshape(-1, 1)
        return -np.mean(diff)
                 
    # gradient function
    def gradient(self, X, y, yhat):
        # initialize the gradient as a zero vector
        grad = np.zeros(self.beta.shape)

        # compute delta J/delta beta for each j
        for j in range(X.shape[1]):
            first = np.multiply(yhat - y, X[:, j][: ,np.newaxis])
            grad[j] = np.mean(first)
        
        return grad
    
    # traini gradient descent
    def fit(self, X, y):
        """
        Args:
            X (numpy array): training data, dimension m x n
            y (numpy array): testing data, dimension m x 1

        Returns:
            beta (numpy array): the final weight
            loss (float): the final cost
        """
        m, n = X.shape
        self.beta = np.zeros((n+1, 1))    # +1 accounts for constant coefficient beta_0
        X = np.hstack((np.ones((m, 1)), X))    # add a constant column for constant coefficient
        losses = []

        # training loop
        for epoch in range(self.epochs):
            yhat = y_hat(self.beta, X)
            grad = gradient(self.beta, X, y, yhat)

            # update beta and loss
            self.beta -= self.alpha*grad
            loss = cost(y, yhat)
            self.losses.append(loss)

            # log result of each epoch
            logging.info("*"*50)
            logging.info(f'epoch = {epoch}')
            logging.info(f'beta = {self.beta}')
            logging.info(f'grad = {grad}')
            #logging.info(f'yhat = {yhat[:50]}')
            #logging.info(f'y = {y[:50]}')
            logging.info(f'loss = {loss}')
                 
                 
    # use trained beta to predict new data
    def predict(self, X):
        """
        Args:
            X (numpy array): feature to predict

        Returns:
            y (numpy array): predict results (0 or 1)
        """
        m = X.shape[0]
        X = np.hstack((np.ones((m, 1)), X))    # add constant column to X for intercept
        pred_proba = sigmoid(np.dot(X, self.beta)).ravel()
        # print(pred_proba.shape)
        pred = np.array([round(x) for x in pred_proba])
        return pred, pred_proba

