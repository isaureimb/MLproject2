from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Le pipeline ajuste automatiquement le StandardScaler UNIQUEMENT sur X_train
# (et sur chaque fold d'entraînement si tu fais du K-Fold), puis transforme X_val / X_test :
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

model.fit(X_train, y_train_mean)
y_val_pred = model.predict(X_val)