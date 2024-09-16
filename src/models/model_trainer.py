from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import time

def train_linear_regression(X_train, y_train):
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def train_decision_tree(X_train, y_train):
    start_time = time.time()
    model = DecisionTreeRegressor(random_state=12345)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,       # Reducir el número de árboles
        max_depth=10,           # Limitar la profundidad máxima de los árboles
        max_features='sqrt',    # Reducir el número de características consideradas
        n_jobs=-1,              # Paralelizar el entrenamiento (usa todos los núcleos disponibles)
        random_state=42
    )
    start_time = time.time()
    model.fit(X_train, y_train)
     # Medir el tiempo de entrenamiento
    training_time_forest = time.time() - start_time
    return model, training_time_forest

def train_lightGBM(X_train, y_train, X_valid, y_valid):
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1
    }
    
    start_time = time.time()
    lgbm_reg = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(100),
        ],
    )
    training_time = time.time() - start_time
    return lgbm_reg, training_time

def predict_model(model, X_test):
    return model.predict(X_test)