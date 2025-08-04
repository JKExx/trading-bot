"""Model training utilities."""
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y) -> RandomForestClassifier:
    """Train a basic random forest classifier."""
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
