"""
Deep Learning Module using TensorFlow/Keras
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Deep learning features disabled.")

class DeepLearningModels:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.history = {}
        
    def create_regression_model(self, input_dim, hidden_layers=[128, 64, 32]):
        """Create a deep neural network for regression"""
        if not TF_AVAILABLE:
            return None
            
        model = models.Sequential([
            layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(hidden_layers[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_layers[2], activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)  # Output layer for regression
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def create_classification_model(self, input_dim, num_classes, hidden_layers=[128, 64, 32]):
        """Create a deep neural network for classification"""
        if not TF_AVAILABLE:
            return None
            
        model = models.Sequential([
            layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(hidden_layers[1], activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(hidden_layers[2], activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model
    
    def train_regression(self, X_train, X_test, y_train, y_test, epochs=100, batch_size=32):
        """Train regression model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return None, None
            
        print(f"\n{'='*60}")
        print("DEEP LEARNING: Regression Model")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        model = self.create_regression_model(X_train_scaled.shape[1])
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train
        print(f"Training on {X_train_scaled.shape[0]} samples...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_train = model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_test = model.predict(X_test_scaled, verbose=0).flatten()
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\nResults:")
        print(f"  Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
        print(f"  Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
        self.models['regression'] = model
        self.history['regression'] = history
        
        return model, history
    
    def train_classification(self, X_train, X_test, y_train, y_test, num_classes=2, epochs=100, batch_size=32):
        """Train classification model"""
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return None, None
            
        print(f"\n{'='*60}")
        print("DEEP LEARNING: Classification Model")
        print(f"{'='*60}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        model = self.create_classification_model(X_train_scaled.shape[1], num_classes)
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        
        # Train
        print(f"Training on {X_train_scaled.shape[0]} samples...")
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        if num_classes == 2:
            y_pred_train = (model.predict(X_train_scaled, verbose=0) > 0.5).astype(int).flatten()
            y_pred_test = (model.predict(X_test_scaled, verbose=0) > 0.5).astype(int).flatten()
        else:
            y_pred_train = np.argmax(model.predict(X_train_scaled, verbose=0), axis=1)
            y_pred_test = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")
        print(f"  Epochs trained: {len(history.history['loss'])}")
        
        self.models['classification'] = model
        self.history['classification'] = history
        
        return model, history

if __name__ == "__main__":
    print("Deep Learning Module Ready")
    print(f"TensorFlow Available: {TF_AVAILABLE}")
