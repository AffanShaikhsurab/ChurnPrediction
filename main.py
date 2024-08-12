import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
import shap
import os
import joblib
from scipy.sparse import issparse


# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    if 'Customer_ID' in data.columns:
        data = data.drop('Customer_ID', axis=1)

    # Handle NaN values in the target variable
    data = data.dropna(subset=['churn'])

    X = data.drop('churn', axis=1)
    y = data['churn']

    # Ensure y is integer type
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    if issparse(X_train_preprocessed):
        X_train_preprocessed = X_train_preprocessed.toarray()
    if issparse(X_test_preprocessed):
        X_test_preprocessed = X_test_preprocessed.toarray()

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor

# 2. Feature Engineering
@tf.function
def feature_engineering(X):
    X_new = tf.concat([
        X,
        tf.expand_dims(X[:, 0] * X[:, 1], axis=1),
        tf.expand_dims(tf.reduce_sum(X[:, :3], axis=1), axis=1),
        tf.expand_dims(tf.reduce_mean(X[:, 3:], axis=1), axis=1)
    ], axis=1)
    return X_new

# 3. LSTM with Attention
def build_lstm_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(32, return_sequences=True)(inputs)  # Reduced from 64
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)  # Reduced from 8 heads, 64 key_dim
    x = LayerNormalization()(x)
    x = Dense(32, activation='relu')(x)  # Reduced from 64
    x = Dropout(0.2)(x)  # Reduced from 0.3
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# 4. Base Models
def train_base_models(X_train, y_train):
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        max_depth=4,  # Reduced from 6
        learning_rate=0.1,
        n_estimators=100,  # Reduced from 1000
        tree_method='hist',
        device='cpu'  # Changed to CPU for compatibility
    )
    xgb_model.fit(X_train, y_train)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,  # Reduced from 1000
        device='cpu'  # Changed to CPU for compatibility
    )
    lgb_model.fit(X_train, y_train)

    # Neural Network
    nn_model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Reduced from 256
        Dropout(0.2),  # Reduced from 0.3
        Dense(64, activation='relu'),  # Reduced from 128
        Dropout(0.2),  # Reduced from 0.3
        Dense(32, activation='relu'),  # Reduced from 64
        Dropout(0.2),  # Reduced from 0.3
        Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=512, validation_split=0.2,  # Reduced epochs from 100, increased batch size
                 callbacks=[EarlyStopping(patience=5), ReduceLROnPlateau(patience=3)])

    # SVM (with reduced sample size for faster computation)
    sample_size = min(10000, X_train.shape[0])  # Use at most 10,000 samples
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]
    svm_model = SVC(kernel='rbf', C=1.0, probability=True)
    svm_model.fit(X_train_sample, y_train_sample)

    return xgb_model, lgb_model, nn_model, svm_model, xgb_model.predict_proba(X_train)[:, 1], lgb_model.predict_proba(X_train)[:, 1], nn_model.predict(X_train).flatten(), svm_model.predict_proba(X_train)[:, 1]

# 5. Adaptive Ensemble
def build_adaptive_ensemble(base_models, meta_features_shape):
    base_inputs = [Input(shape=(1,)) for _ in range(len(base_models))]
    meta_features_input = Input(shape=meta_features_shape)

    concat = tf.keras.layers.concatenate(base_inputs + [meta_features_input])
    x = Dense(32, activation='relu')(concat)  # Reduced from 64
    x = Dense(16, activation='relu')(x)  # Reduced from 32
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_inputs + [meta_features_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 6. Online Learning Component
class OnlineLearning(tf.keras.callbacks.Callback):
    def __init__(self, data_generator):
        super(OnlineLearning, self).__init__()
        self.data_generator = data_generator

    def on_batch_end(self, batch, logs=None):
        X_online, y_online = next(self.data_generator)
        self.model.train_on_batch(X_online, y_online)

# 7. Transfer Learning (simplified)
def apply_transfer_learning(model, new_data):
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.fit(new_data[0], new_data[1], epochs=5, batch_size=512)  # Reduced epochs from 10
    return model

# 8. Interpretability
def interpret_model(model, X):
    # Create a new model that outputs the last timestep
    last_timestep_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # Use the new model for SHAP explanation
    explainer = shap.DeepExplainer(last_timestep_model, X[:100])  # Reduced sample size for SHAP
    shap_values = explainer.shap_values(X[:100])
    return shap_values

# Main execution
if __name__ == "__main__":
    # Use GPU if available
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Create a directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('telco_customer_churn.csv')

    # Save preprocessor
    joblib.dump(preprocessor, 'checkpoints/preprocessor.joblib')

    # Feature engineering
    X_train_eng = feature_engineering(tf.constant(X_train, dtype=tf.float32)).numpy()
    X_test_eng = feature_engineering(tf.constant(X_test, dtype=tf.float32)).numpy()

    # Reshape data for LSTM
    X_train_lstm = X_train_eng.reshape((X_train_eng.shape[0], 1, X_train_eng.shape[1]))
    X_test_lstm = X_test_eng.reshape((X_test_eng.shape[0], 1, X_test_eng.shape[1]))

    # Build and train LSTM with Attention
    lstm_model = build_lstm_attention((1, X_train_eng.shape[1]))
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    lstm_checkpoint = ModelCheckpoint('checkpoints/lstm_model.keras', save_best_only=True, monitor='val_loss')

    lstm_model.fit(X_train_lstm, y_train,
                   epochs=50, batch_size=512, validation_split=0.15,  # Reduced epochs from 100, increased batch size
                   callbacks=[EarlyStopping(patience=5), lstm_checkpoint, ReduceLROnPlateau(patience=3)])

    # Train base models
    xgb_model, lgb_model, nn_model, svm_model, xgb_preds, lgb_preds, nn_preds, svm_preds = train_base_models(X_train_eng, y_train)

    # Save base models
    joblib.dump(xgb_model, 'checkpoints/xgb_model.joblib')
    joblib.dump(lgb_model, 'checkpoints/lgb_model.joblib')
    nn_model.save('checkpoints/nn_model.keras')

    lstm_model.save('checkpoints/lstm_model_transfer.keras')
    joblib.dump(svm_model, 'checkpoints/svm_model.joblib')

    # Prepare inputs for adaptive ensemble
    base_preds = [
        xgb_model.predict_proba(X_train_eng)[:, 1],
        lgb_model.predict_proba(X_train_eng)[:, 1],
        nn_model.predict(X_train_eng).flatten(),
        svm_model.predict_proba(X_train_eng)[:, 1]
    ]
    meta_features = np.column_stack(base_preds)

    # Build and train adaptive ensemble
    adaptive_ensemble = build_adaptive_ensemble([xgb_model, lgb_model, nn_model, svm_model], meta_features.shape[1:])

    ensemble_checkpoint = ModelCheckpoint('checkpoints/adaptive_ensemble.keras', save_best_only=True, monitor='val_loss')

    adaptive_ensemble.fit([p.reshape(-1, 1) for p in base_preds] + [meta_features], y_train,
                          epochs=50, batch_size=512, validation_split=0.2,  # Reduced epochs from 100, increased batch size
                          callbacks=[ensemble_checkpoint, ReduceLROnPlateau(patience=3)])

    # Online learning (simulated)
def data_generator():
    while True:
        idx = np.random.randint(0, X_train_eng.shape[0], 32)
        yield [p[idx].reshape(-1, 1) for p in base_preds] + [meta_features[idx]], y_train.iloc[idx]

    online_learning_callback = OnlineLearning(data_generator())
    adaptive_ensemble.fit([p.reshape(-1, 1) for p in base_preds] + [meta_features], y_train,
                          epochs=5, callbacks=[online_learning_callback])  # Reduced epochs from 10

    # Save updated adaptive ensemble
    adaptive_ensemble.save('checkpoints/adaptive_ensemble_online.keras')

    # Transfer learning (simulated with a subset of data)
    new_data = (X_test_lstm[:100], y_test[:100])
    lstm_model = apply_transfer_learning(lstm_model, new_data)

    # Save transfer learned model
    lstm_model.save('checkpoints/lstm_model_transfer.keras')

    # Model interpretation (using a subset of data)
    shap_values = interpret_model(lstm_model, X_test_lstm[:1000])

    print("Model training and evaluation completed.")

    # Evaluate final model
    ensemble_preds = adaptive_ensemble.predict([p.reshape(-1, 1) for p in [
        xgb_model.predict_proba(X_test_eng)[:, 1],
        lgb_model.predict_proba(X_test_eng)[:, 1],
        nn_model.predict(X_test_eng).flatten(),
        svm_model.predict_proba(X_test_eng)[:, 1]
    ]] + [np.column_stack([
        xgb_model.predict_proba(X_test_eng)[:, 1],
        lgb_model.predict_proba(X_test_eng)[:, 1],
        nn_model.predict(X_test_eng).flatten(),
        svm_model.predict_proba(X_test_eng)[:, 1]
    ])])

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = (ensemble_preds > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))