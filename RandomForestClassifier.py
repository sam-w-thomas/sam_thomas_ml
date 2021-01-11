from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union 
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_X_y
import random
import sklearn
from matplotlib import pyplot as plt
from sklearn.utils.validation import check_is_fitted
import numpy as np
from math import exp
from math import log
import warnings
import utilities as util # provides general classifier functionality, programmed by Samuel Thomas

warnings.simplefilter(action='ignore', category=FutureWarning)

class RandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 size: int=None, 
                 max_depth: int=None, 
                 max_features: Union[int, str]="sqrt",
                 criteria: str="gini", 
                 splitter: str="best", 
                 min_samples_split: int=2, 
                 min_samples_leaf: int=1,
                 max_leaf_nodes: int=None,
                 bootstrap_num: Union[float,int]=0.25,
                 method: str="bagging"
                ):
        """
        Create Random Forest model
        
        Args:
            size: Number of estimators in RandomForest
            max_depth: Maximum depth of Decision Trees
            max_features: Sets maximum number of features per Decision Tree. Introduces feature variance
            criteria: Splitting method used
            splitter: 
            min_samples_split:
            max_samples_leaf:
            max_leaf_nodes:
            bootstrap_num
            method: Set ensemble technique, Bagging, Boosting or Stacking [str]
        """
        self.size = size
        self.max_depth = max_depth
        self.max_features = max_features
        self.criteria = criteria
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap_num = bootstrap_num
        self.method = method
        self._sample_size = 0
    
    
    def fit(self,
            data: np.ndarray,
            target: Union[np.ndarray, None]
           ):
        """
        Build classifier (train the classifier)
        
        Args:
            data: Feature-Matrix used to train classifier
            target: Target values for classifier
            
        Raises:
            ValueError 
            TypeError
        """
        data, target = check_X_y(data,target)
        self._sample_size = len(data)
        self._target_type = target.dtype
        
        if self.method=="bagging":
            self.build_forest(data,target)
        elif self.method=="boosting":
            self.build_forest_boost(data,target)
        else:
            raise ValueError("Method of " + self.method + " is not a valid method")
        
        self.fitted_ = True
        
        
    def build_forest(self,
                     data,
                     target
                    ):
        """
        Builds forest - called by .fit
        
        Args:
            data: Feature-Matrix 
            target: Target classification
        
        Raises:
        """
        trees = []
        
        data = data
        target = target
        size = self.size 
        
        for _ in range(size):
            tree = DecisionTreeClassifier(criterion=self.criteria, 
                                          splitter=self.splitter, 
                                          max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf,
                                          max_leaf_nodes=self.max_leaf_nodes,
                                          max_features=self.max_features
                                         )
            
            data_bootstrap, target_bootstrap = self.bootstrap(data,target,self.bootstrap_num)
            
            tree.fit(data_bootstrap,target_bootstrap)
            trees.append(tree)
            
        self.trees = trees
        
    def build_forest_boost(self,
            data: np.ndarray,
            target: np.ndarray
           ):
        """
        Builds (Ada)boosted forest - called by .fit
        
        Args:
            data: Feature-matrix [Matrix]
            target: Target classification [Array] 
        
        Raises
        """
        target, self._labels = util.binary_encode(target, self) #AdaBoost is binary classification by default. Could  implment multi-class AdaBoost via AdaBoost-SAMME (Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009)

        self.trees = []
        
        for i in range(self.size):
            tree = DecisionTreeClassifier(
                            max_depth=self.max_depth   #Create stump (weak learner)
                            )
            tree.fit(data,target)

            predicted = tree.predict(data)
            
            #Calculate model peformance and error
            correct = util.correct_class(target,predicted, self._sample_size)
            weights = np.full(self._sample_size, fill_value=1/self._sample_size, dtype=float)
            errors = np.array([weights[i] for i in range(self._sample_size) if correct[i] == 0])
            total_error = np.sum(errors) 
            
            if total_error == 0.0: #Perfect fit when all predictions correct 
                print("AdaBoost hit a perfect fit after" + str(int(len(self.trees)) + 1) + "iterations")
                break
               
            model_perf = 0.5 * log((1 - total_error)/ total_error)
            
            self.trees.append([tree, model_perf]) #Add tree and model performance forest
            
            #Update weights
            for i in range(self._sample_size):
                if correct[i] == 0:
                    weights[i] = weights[i] * np.exp(model_perf)
                else:
                    weights[i] = weights[i] * np.exp(-(model_perf))
            weights = weights/np.sum(weights)

            #Get new sample distribution
            rows = [value for value in range(self._sample_size)]
            sel_rows = np.random.choice(rows,size=self._sample_size, replace=True, p=weights)
 
            data = data[[sel_rows]] 
            target = target[[sel_rows]]
        
        self.trees = np.array(self.trees)
            
         
    def predict(self,
                data: np.ndarray
               ):
        """
        Predict classes for data supplied. Columns are treated as the vote for each sample.
        
        Args:
            data: Feature-Matrix to classify
            
        Returns:
            Results of classification. Array structure
            
        Raises:
            NotFittedError
        
        """
        check_is_fitted(self)
        
        if self.method=="bagging":
            result = self.predict_bag(data)
        elif self.method=="boosting":
            result = self.predict_boost(data)
        else:
            raise ValueError("Method of " + self.method + " is not a valid method")
            
        return np.array(result,dtype=self._target_type) # Make same type as expected
        
        
    def predict_bag(self, data):
        """
        Predicts classes using (bagging) forest - called by .predict
        
        Args:
            target: Array structure to classify
        
        Raises:
            NotFittedError
        """
        
        tree_size = len(self.trees)
        results = np.empty((tree_size, self._sample_size), dtype=object)
        # Matrix containg classification for all trees - column represents vote for each sample
        for i in range(tree_size):
            result = self.trees[i].predict(data) 
            results[i] = result 
            
        return util.calc_vote(results) # Majority vote calculation
    
    
    def predict_boost(self,
                data: np.ndarray
               ):
        """
        Predicts classes using boosted forest - called by .predict
        
        Args:
            data: Feature-Matrix to classify
            
        Returns:
            Results of classification. Array structure
            
        Raises:
            NotFittedError
        
        """
        sum = 0
        
        result_matrix = np.empty(shape=(self._sample_size, self.size))
        
        # Create matrix of results - each row is predicted values for each sample
        column_i = 0
        for tree in self.trees:
            model = tree[0]
            check_is_fitted(model)
            result_matrix[:,column_i] = model.predict(data)
            column_i += 1
        
        # Calculate final result
        prediction = np.empty(self._sample_size, dtype=int)
        for i in range(self._sample_size):
            sum_val = 0
            for j in range(self.size):
                sum_val += self.trees[j][1] * result_matrix[i][j]
                
            prediction[i] = np.sign(sum_val)
        
        return util.binary_decode(prediction,self)
        
            
    def bootstrap(self, 
                  data, 
                  target, 
                  bootstrap_num
                 ):
        """
        Generate random subset of instances(sampels) - performs row sampling with replacement 
        
        Args:
            bootstrap_num: Number of instances (samples) to return. If Int - specific number of instances, If Float - percentage out of data
            
        Return: 
            data: Bootstrapped Feature matrix
            target: Bootstrapped Labels
        
        """
        
        # Check bootstrap_num is correct
        if isinstance(bootstrap_num, float):
            if not 0 < bootstrap_num < 1:
                raise ValueError   
            bootstrap_num = int(len(data) * bootstrap_num)
            
        elif isinstance(bootstrap_num, int):
            if bootstrap_num > len(data):
                raise ValueError
        else:
            raise ValueError
            
        sel_row = random.sample(range(0, len(data)), bootstrap_num)

        data = data[sel_row]
        target = target[sel_row]
        
        return data, target
    
    def score(self,
              data,
              target
             ):
        """
        Calculate mean accuracy score for data and target labels
        
        Args:
            data: Feature-matrix
            target: Target classification labels
            
        Raises:
            NotFittedError
        """
        check_is_fitted(self)
        predicted = self.predict(data)
        return sklearn.metrics.accuracy_score(target,predicted)
              
    
    def show_tree(self,
                  tree_index: int,
                  size: tuple=(15,15)
                 ) -> None:
        """
        Shows logic behind decision tree in ensemble
        
        Args:
            tree_index: Position (0-index) of tree in Ensemble. Must be less than size [int]
            size: Size of decision tree pyplot [tuple]
        
        Raises:
            ValueError
        """
        
        if tree_index > len(self.trees) - 1 or not int:
            raise ValueError
            
        plt.figure(figsize=size)
        if len(self.trees[0]) == 2:
            sel_tree = self.trees[tree_index][0]
        else:
            sel_tree = self.trees[tree_index]
        sklearn.tree.plot_tree(sel_tree)
        plt.show()
        
        
    def get_trees(self):
        """
        Get all trees in ensemble
        
        Returns:
            All decisions tree in ensmble. Returns None if .fit not called
        """
        check_is_fitted(self)
        return self.trees      