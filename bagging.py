
# methods/bagging.py
from sklearn.ensemble import BaggingClassifier
import numpy as np
from .abstract import AbstractClass
import logging

class Bagging(AbstractClass):
    """
    A wrapper class for sklearn's BaggingClassifier with additional functionality.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the BaggingClassifier with provided keyword arguments.
        
        Args:
            **kwargs: Keyword arguments to pass to BaggingClassifier
        """
        # Convert base_estimator to estimator if present in kwargs
        if 'base_estimator' in kwargs:
            kwargs['estimator'] = kwargs.pop('base_estimator')
        
        self.model = BaggingClassifier(**kwargs)
    
    def predict(self, input_data):
        """
        Make predictions using the trained model.
        
        Args:
            input_data: Array-like of shape (n_samples, n_features)
            
        Returns:
            Array-like of shape (n_samples,): Predicted classes
        """
        try:
            return self.model.predict(input_data)
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
    
    def fit(self, dataset):
        """
        Fit the bagging classifier to the training data.
        
        Args:
            dataset: Tuple of (X, y) where X is the feature matrix and y is the target vector
        """
        try:
            X, y = dataset
            self.model.fit(X, y)
        except Exception as e:
            logging.error(f"Error in fitting model: {str(e)}")
            raise
    
    def predict_proba(self, input_data):
        """
        Predict class probabilities for input data.
        
        Args:
            input_data: Array-like of shape (n_samples, n_features)
            
        Returns:
            Array-like of shape (n_samples, n_classes): Predicted class probabilities
        """
        try:
            return self.model.predict_proba(input_data)
        except Exception as e:
            logging.error(f"Error in probability prediction: {str(e)}")
            raise
    
    def get_feature_importance(self):
        """
        Get feature importance if the base estimator supports it.
        
        Returns:
            Array of feature importance scores if available, None otherwise
        """
        try:
            if hasattr(self.model.estimator_, 'feature_importances_'):  # Changed from base_estimator_ to estimator_
                return self.model.estimator_.feature_importances_
            return None
        except Exception as e:
            logging.error(f"Error getting feature importance: {str(e)}")
            return None