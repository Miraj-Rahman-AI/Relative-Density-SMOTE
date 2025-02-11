import random
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
from sklearn.cluster import k_means
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import copy

# Load the Iris dataset
iris = load_iris()

# Extract features (X) and labels (y)
X = iris.data
y = iris.target

def number_maj(imbalanced_featured_data, minor_feature_data, minor_label, imbalanced_label_data):
    """
    Calculate how many majority samples are in the k nearest neighbors of the minority samples.
    
    :param imbalanced_featured_data: Feature data of the entire dataset.
    :param minor_feature_data: Feature data of the minority class.
    :param minor_label: Label of the minority class.
    :param imbalanced_label_data: Label data of the entire dataset.
    :return: Number of majority samples in the k nearest neighbors of each minority sample.
    """
    nnm_x = NearestNeighbors(n_neighbors=6).fit(imbalanced_featured_data).kneighbors(minor_feature_data, return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != minor_label).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return n_maj

class RSmote:
    """
    RSmote class for handling imbalanced datasets by generating synthetic samples for the minority class.
    """
    
    def __init__(self, data, ir=1, k=5, random_state=None):
        """
        Initialize the RSmote class.
        
        :param data: Dataset with labels in the 0th column.
        :param ir: Imbalanced ratio for synthetic data generation.
        :param k: Number of nearest neighbors to consider.
        :param random_state: Seed for random number generation.
        """
        self.data = data  # Labels are in the first column
        self._div_data()
        self.n_train_less, self.n_attrs = self.train_less.shape
        self.IR = ir
        self.k = k
        self.new_index = 0
        self.random_state = random_state
        self.N = 0
        self.synthetic = None

    def _div_data(self):
        """
        Divide the dataset into majority and minority classes.
        """
        count = Counter(self.data[:, 0])
        a, b = set(count.keys())
        self.tp_less, self.tp_more = (a, b) if count[a] < count[b] else (b, a)

        data_less = self.data[self.data[:, 0] == self.tp_less]
        data_more = self.data[self.data[:, 0] == self.tp_more]

        self.train_less = data_less
        self.train_more = data_more

        self.train = np.vstack((self.train_more, self.train_less))

    def over_sampling(self):
        """
        Perform oversampling of the minority class.
        
        :return: Resampled dataset with synthetic samples.
        """
        if self.k + 1 > self.n_train_less:
            print('Expected n_neighbors <= n_samples,  but n_samples = {}, n_neighbors = {}, '
                  'has changed the n_neighbors to {}'.format(self.n_train_less, self.k + 1, self.n_train_less))
        
        data_less_filter = []
        num_maj_filter = []
        length_less = len(self.train_less)
        num_maj = number_maj(self.train[:, 1:], self.train_less[:, 1:], self.tp_less, self.train[:, 0])
        for m in range(len(num_maj)):
            if num_maj[m] < self.k:
                data_less_filter.append(self.train_less[m])
                num_maj_filter.append(num_maj[m])
        self.train_less = np.array(data_less_filter)
        if self.k + 1 > len(self.train_less):
            self.k = len(self.train_less) - 1  # Ensure the number of samples is greater than the number of neighbors

        distance_more, nn_array_more = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_more[:, 1:]).kneighbors(
            self.train_less[:, 1:], return_distance=True)

        distance_less, nn_array = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_less[:, 1:]).kneighbors(
            self.train_less[:, 1:], return_distance=True)

        distance_less = distance_less.sum(axis=1)
        distance_more = distance_more.sum(axis=1)
        distance = distance_less / distance_more
        density = 1 / distance  # Calculate density

        density = list(map(lambda x: min(100, x), density))  # Control the maximum density range at 100

        # Sort the density and minority samples in order of density
        density_sorted = sorted(range(len(density)), key=lambda a: density[a], reverse=True)
        data_resorted = []
        density_sorted_data = []
        num_sorted = []
        for i in range(len(self.train_less)):
            data_resorted.append(self.train_less[density_sorted[i]])
            density_sorted_data.append(density[density_sorted[i]])
            num_sorted.append(num_maj_filter[density_sorted[i]])

        density = np.array(density_sorted_data)
        cluster_big_density = []
        cluster_small_density = []
        cluster_big_data = []
        cluster_small_data = []
        cluster_big_num = []
        cluster_small_num = []
        cluster = k_means(X=density.reshape((len(density), 1)), n_clusters=2)
        for i in range(cluster[1].shape[0]):
            if cluster[1][i] != cluster[1][i + 1]:  # Partition cluster
                cluster_big_density = density[:i + 1]
                cluster_big_data = np.array(data_resorted)[:i + 1, :]
                cluster_big_num = num_sorted[:i + 1]
                cluster_small_density = density[i + 1:]
                cluster_small_data = np.array(data_resorted)[i + 1:, :]
                cluster_small_num = num_sorted[i + 1:]
                break

        # If there is only one point in a cluster, do not divide the cluster
        if len(cluster_big_data) < 2 or len(cluster_small_data) < 2:
            cluster_big_data = np.array(data_resorted)
            cluster_big_density = density
            cluster_big_num = num_sorted
            flag = 1  # If flag==1, only run the big cluster once
        else:
            flag = 2
        sum_0 = 0
        sum_1 = 0

        # Calculate weight
        for p in range(len(cluster_big_num)):
            sum_0 += (5 - cluster_big_num[p]) / self.k + 1
        for p in range(len(cluster_small_num)):
            sum_1 += (5 - cluster_small_num[p]) / self.k + 1

        ratio = []  # Save the total weight of each cluster
        ratio.append(sum_0)
        ratio.append(sum_1)
        wight = [5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]
        kk = self.k
        diff = len(self.train_more) - length_less  # The number of samples needed to synthesize
        totol_less = len(self.train_less)

        for i in range(flag):
            if i == 0:  # Big cluster
                density = cluster_big_density
                self.n_train_less = len(cluster_big_data)
                self.train_less = cluster_big_data
                maj_num_ab = cluster_big_num
            else:  # Small cluster
                density = cluster_small_density
                self.n_train_less = len(cluster_small_data)
                self.train_less = cluster_small_data
                maj_num_ab = cluster_small_num

            self.k = min(len(self.train_less) - 1, kk)  # If len(self.train_less) < k, set k = len(self.train_less)

            # The number of sample points that need to be inserted at each point
            if flag == 1:
                number_synthetic = int(len(self.train_more) / self.IR - len(self.train_less))
            else:
                if i == 0:
                    number_synthetic = int((len(self.train_less) / totol_less) * diff)
                    len_big = number_synthetic
                else:
                    number_synthetic = diff - len_big

            # Calculate how many points should be inserted for each sample
            N = list(map(lambda x: int((x / ratio[i]) * number_synthetic), wight))
            self.reminder = number_synthetic - sum(N)
            self.num = 0

            neighbors = NearestNeighbors(n_neighbors=self.k + 1).fit(self.train_less[:, 1:])
            nn_array = neighbors.kneighbors(self.train_less[:, 1:], return_distance=False)

            self.synthetic = np.zeros((number_synthetic, self.n_attrs - 1))
            for p in range(self.train_less.shape[0]):
                self._populate(p, nn_array[p][1:], number_synthetic, N, maj_num_ab)

            label_synthetic = np.array([self.tp_less] * number_synthetic).reshape((number_synthetic, 1))
            np.random.seed(self.random_state)
            synthetic_dl = self.synthetic
            synthetic_dl = np.hstack((label_synthetic, synthetic_dl))  # Add class column

            data_res = synthetic_dl
            if i == 0:
                return_data = np.vstack((copy.deepcopy(self.train), data_res))
                if flag == 1:
                    return return_data
                self.new_index = 0
            else:
                return_data = np.vstack((copy.deepcopy(return_data), data_res))

                return return_data

    def _populate(self, index, nnarray, number_synthetic, N, maj_num_ab):
        """
        Generate synthetic samples for the minority class.
        
        :param index: Index of the current minority sample.
        :param nnarray: Nearest neighbors of the current minority sample.
        :param number_synthetic: Total number of synthetic samples to generate.
        :param N: Number of synthetic samples to generate for each minority sample.
        :param maj_num_ab: Number of majority samples in the nearest neighbors.
        """
        random.seed(self.random_state)
        if self.num < self.reminder:
            turn = N[maj_num_ab[index]] + 1
        else:
            turn = N[maj_num_ab[index]]
        for j in range(turn):
            if self.new_index < number_synthetic:
                if self.k == 1:
                    nn = 0
                else:
                    nn = random.randint(0, self.k - 1)
                dif = self.train_less[nnarray[nn], 1:] - self.train_less[index, 1:]
                gap = random.random()
                self.synthetic[self.new_index] = self.train_less[index, 1:] + gap * dif
                self.new_index += 1
            else:
                break
        self.num += 1

class RSmoteKClasses:
    """
    RSmoteKClasses class for handling multi-class imbalanced datasets.
    """
    
    def __init__(self, ir=1, k=5, random_state=None):
        """
        Initialize the RSmoteKClasses class.
        
        :param ir: Imbalanced ratio for synthetic data generation.
        :param k: Number of nearest neighbors to consider.
        :param random_state: Seed for random number generation.
        """
        self.ir = ir
        self.k = k
        self.random_state = random_state

    def fit_resample(self, X, y):
        """
        Resample the dataset to balance the classes.
        
        :param X: Feature matrix.
        :param y: Label vector.
        :return: Resampled feature matrix and label vector.
        """
        data = np.hstack((y.reshape((len(y), 1)), X))
        counter = Counter(y)

        max_class_label, max_class_number = 0, 0
        for k, v in counter.items():
            if v > max_class_number:
                max_class_label, max_class_number = k, v

        data_new = np.array([]).reshape((-1, data.shape[1]))

        data_more = data[data[:, 0] == max_class_label, :]
        for k, v in counter.items():
            if v == max_class_number:
                continue
            data_less = data[data[:, 0] == k, :]
            data_train = np.vstack((data_more, data_less))
            r_smote = RSmote(data_train, random_state=self.random_state)
            data_r_smote = r_smote.over_sampling()
            if data_new.shape[0] == 0:
                data_new = np.vstack((data_new, data_r_smote))
            else:
                data_new = np.vstack((data_new, data_r_smote[data_r_smote[:, 0] != max_class_label, :]))

        X_resampled, y_resampled = data_new[:, 1:], data_new[:, 0]

        return X_resampled, y_resampled
    
    def visualize_data(self, X, y, title="Dataset Visualization"):
        """
        Visualize the dataset.
        
        :param X: Feature matrix.
        :param y: Label vector.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        unique_classes = np.unique(y)
        
        for class_label in unique_classes:  # Fixed syntax error
            class_indices = y == class_label
            plt.scatter(X[class_indices, 0], X[class_indices, 1], label=f"Class {class_label}", alpha=0.7)
        
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True)
        plt.show()

# Ensure correct indentation for script execution
if __name__ == '__main__':
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Visualize the original dataset
    print("Original X values:")
    print(X)
    print("\nOriginal y values:")
    print(y)

    # Create instance of RSmoteKClasses
    rsmotek = RSmoteKClasses(ir=1, k=5, random_state=42)
    
    # Resample the dataset
    X_resampled, y_resampled = rsmotek.fit_resample(X, y)

    # Output the resampled data
    print("Resampled X values:")
    print(X_resampled)
    print("\nResampled y values:")
    print(y_resampled)

    # Visualize the datasets
    rsmotek.visualize_data(X[:, :2], y, title="Original Dataset")
    rsmotek.visualize_data(X_resampled[:, :2], y_resampled, title="Resampled Dataset")