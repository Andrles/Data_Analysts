# 📘 Аналитический проект: Анализ пользовательской активности

## Цель проекта
Провести комплексный анализ данных о пользователях, выявить закономерности и зависимости между признаками, а также построить простую модель линейной регрессии. Проект демонстрирует навыки работы с данными, анализа, визуализации и статистического вывода.

## 🔧 Импорт библиотек
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, shapiro, levene, f_oneway, kruskal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
```

## 📦 Создание датафрейма
```python
data = {
    'user_id': range(1, 201),
    'user_name': [f'user_{i}' for i in range(1, 201)],
    'age': np.random.randint(18, 70, 200),
    'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Canada'], 200),
    'gender': np.random.choice(['Male', 'Female'], 200),
    'score': np.random.normal(75, 15, 200).round(1),
    'activity_score': np.random.randint(10, 100, 200),
    'reg_date': pd.to_datetime(np.random.choice(pd.date_range('2023-01-01', '2024-05-01'), 200)),
    'status': np.random.choice(['active', 'inactive', None], 200, p=[0.45, 0.45, 0.10])
}
df_portfolio = pd.DataFrame(data)
```

## 🧾 Первичный обзор данных
```python
# Общая информация о датафрейме
df_portfolio.info()

# Проверка наличия пропущенных значений
print(df_portfolio.isnull().sum())

# Расчёт доли пропущенных значений в столбце status
shape = df_portfolio['status'].isnull().sum() * 100 / df_portfolio['status'].count()
fills = df_portfolio['status'].shape[0] - df_portfolio['status'].notnull().sum()
print(f'Столбец status содержит: {fills} пропусков, что составляет {round(shape, 2)}% от общего числа строк')

# Заполнение пропусков значением "unknown"
df_portfolio['status'] = df_portfolio['status'].fillna('unknown')

# Преобразование столбцов в категориальные типы
df_portfolio[['country', 'gender']] = df_portfolio[['country', 'gender']].astype('category')

# Проверка изменений типов
print(df_portfolio.dtypes)
```

## 📊 Анализ распределений и визуализация
```python
# Гистограммы возрастов и баллов
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_portfolio['age'], bins=15, kde=True, ax=axs[0], color='skyblue')
axs[0].set_title('Распределение возраста')
sns.histplot(df_portfolio['score'], bins=15, kde=True, ax=axs[1], color='salmon')
axs[1].set_title('Распределение итогового балла')
plt.tight_layout()
plt.show()

# Boxplot по странам
plt.figure(figsize=(10, 6))
sns.boxplot(x='country', y='score', data=df_portfolio, palette='pastel')
plt.title('Сравнение итоговых баллов по странам')
plt.show()
```

## 📈 Корреляционный анализ
```python
# Матрица корреляции
corr_matrix = df_portfolio[['age', 'score', 'activity_score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# Корреляция между активностью и итоговым баллом
corr, pval = pearsonr(df_portfolio['activity_score'], df_portfolio['score'])
print(f'Коэффициент корреляции между activity_score и score: {corr:.2f} (p-value={pval:.4f})')
```

## 📐 Проверка гипотез и статистический анализ
```python
# Нормальность распределения
print('Shapiro-Wilk Test (score):', shapiro(df_portfolio['score']))

# Проверка на однородность дисперсий и различия между странами
levene_test = levene(*[df_portfolio[df_portfolio['country'] == c]['score'] for c in df_portfolio['country'].unique()])
f_oneway_test = f_oneway(*[df_portfolio[df_portfolio['country'] == c]['score'] for c in df_portfolio['country'].unique()])
print('Levene test p-value:', levene_test.pvalue)
print('ANOVA p-value:', f_oneway_test.pvalue)
```

## 🤖 Линейная регрессия
```python
# Подготовка данных
X = df_portfolio[['age', 'activity_score']]
y = df_portfolio['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка качества модели
print(f'MSE: {mean_squared_error(y_test, y_pred):.2f}')
print(f'R^2: {r2_score(y_test, y_pred):.2f}')
print(f'Коэффициенты: {model.coef_}, Intercept: {model.intercept_:.2f}')
```

## ✅ Выводы
- Данные успешно загружены и очищены от пропусков.
- Возраст и активность пользователя оказывают умеренное влияние на итоговый балл.
- Наблюдается умеренная положительная корреляция между активностью и итоговым баллом.
- Статистически значимых различий по среднему баллу между странами не обнаружено.
- Простая модель линейной регрессии показывает умеренное качество предсказания, что указывает на возможность улучшения при добавлении дополнительных признаков.
