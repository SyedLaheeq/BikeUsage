import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from scipy.stats import uniform, randint

class XGBoostModel:
    """XGBoost model wrapper with additional functionality for training, evaluation, and tuning."""
    
    def __init__(self, params=None):
        """Initialize XGBoost model with default or custom parameters."""
        self.params = params if params else {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'random_state': 42,
            'enable_categorical': False
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.best_params_ = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with validation data."""
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, 
                y_train,
                eval_set=eval_set,
                callbacks=[xgb.callback.EarlyStopping(rounds=10)],
                verbose=True
            )
        else:
            self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """Make predictions."""
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get probability predictions."""
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance with multiple metrics."""
        y_pred = self.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        for metric_name, value in metrics.items():
            logging.info(f"{metric_name.capitalize()}: {value:.4f}")
        
        return metrics
    
    def tune_hyperparameters(self, X_train, y_train):
        """Perform randomized search for hyperparameter tuning."""
        param_distributions = {
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': randint(100, 1000),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 7),
            'gamma': uniform(0, 0.5)
        }
        
        random_search = RandomizedSearchCV(
            estimator=xgb.XGBClassifier(enable_categorical=False),
            param_distributions=param_distributions,
            n_iter=20,
            scoring='accuracy',
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        self.best_params_ = random_search.best_params_
        self.model = random_search.best_estimator_
        
        logging.info(f"Best parameters found: {self.best_params_}")
        logging.info(f"Best cross-validation score: {random_search.best_score_:.4f}")
        
        return self.best_params_
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance."""
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()