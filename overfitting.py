import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a small dataset
np.random.seed(42)

X = np.linspace(0, 10, 30).reshape(-1, 1)

y = 2 * X.ravel() + 5
y += np.random.normal(0, 3, 30)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Compare different model complexities
for degree in [1, 2, 15]:

    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_error = mean_squared_error(
        y_train, train_pred
    )

    test_error = mean_squared_error(
        y_test, test_pred
    )

    print("\nDegree:", degree)
    print("Training Error:", round(train_error, 2))
    print("Testing Error:", round(test_error, 2))

    plt.plot(
        X,
        model.predict(X),
        label=f"Degree {degree}"
    )

plt.scatter(X, y, label="Real Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Underfitting vs Overfitting")
plt.legend()
plt.show()
