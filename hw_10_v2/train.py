import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

DATA_PATH = "realty_data.csv"
TARGET = "price"

# Используем только эти числовые признаки; period не включаем.
FEATURES = ["lat", "lon", "total_square", "rooms", "floor"]

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET]
missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"В файле не найдены столбцы: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

# Оставляем строки с известной ценой.
valid = y.notna()
X = X.loc[valid]
y = y.loc[valid]

if len(X) < 2:
    raise ValueError("Для обучения нужно не менее двух строк с известной ценой.")

model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)

# При достаточном количестве данных оцениваем модель на тестовой выборке.
if len(X) >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    print(f"R² на тестовой выборке: {model.score(X_test, y_test):.3f}")

# Обучаем итоговую модель на всех доступных данных.
model.fit(X, y)

joblib.dump(
    {"model": model, "features": FEATURES},
    "realty_model.joblib"
)

print(f"Модель сохранена. Используемые признаки: {FEATURES}")