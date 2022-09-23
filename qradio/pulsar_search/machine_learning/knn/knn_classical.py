from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from dataclasses  import dataclass
from knn_data import read_dataset, divide_dataset


class KNN:

    def __init__(self, train_dataset, test_dataset, n_neighbor=3):
        """Handles the knn training/testing

        Args:
            train_dataset (dataclass): train dataset
            test_dataset (dataclass): test dataset
            n_neighbor (int, optional): number of neighbors. Defaults to 3.
        """

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_neighbor = n_neighbor
        self.model =  KNeighborsClassifier(n_neighbors=self.n_neighbor)

    def fit(self):
        """Fit the model
        """
        self.model.fit(self.train_dataset.features, self.train_dataset.labels)

    def test(self):
        """Test the model
        """
        predict = self.model.predict(self.test_dataset.features)
        percent = 100*np.sum(predict == self.test_dataset.labels)/self.test_dataset.npts
        print(" ==> Classification succesfull at %f percent" %percent)




if __name__ == "__main__":

    dataset = read_dataset()
    train_dataset, test_dataset = divide_dataset(dataset)

    knn = KNN(train_dataset, test_dataset)
    knn.fit()
    knn.test()