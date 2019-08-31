
class ModelUtils:

    def __init__(self, config):
        return None

    def train_model(self, learner, X_train, y_train):
        return learner.fit(X_train, y_train)

    def eval_model(self, learner, scorer, X_test, y_test):
        predictions = learner.predict(X_test)
        return scorer(y_test, predictions)


    def save_model(self, model, model_filepath):
        import pickle
        # save the classifier
        pickle.dump(model, open(model_filepath, 'wb'))
        return True

    