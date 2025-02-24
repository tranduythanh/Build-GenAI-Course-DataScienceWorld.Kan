class AI:
  def __init__(self, algorithm, model_type):
    self.algorithm = algorithm
    self.model_type = model_type
    return self

  def fit(self, model, X_train, y_train, predict):
    print("AI.fit ---------------")
    print(f"\tModel:\t\t {model}")
    print(f"\tX_train:\t {X_train}")
    print(f"\ty_train:\t {y_train}")
    print(f"\tPredict:\t {predict}")

  def predict(self, model, X_pred):
    print("AI.predict ---------------")
    print(f"\tModel:\t\t {model}")
    print(f"\tX_pred:\t\t {X_pred}")

class DeepLearning(AI):
  def train_on_epoch(self, model, X_train, y_train, epoch, learning_rate):
    print("DeepLearning.train_on_epoch ---------------")
    print(f"\tModel:\t\t {model}")
    print(f"\tX_train:\t {X_train}")
    print(f"\ty_train:\t {y_train}")
    print(f"\tEpoch:\t\t {epoch}")
  
  def fit(self, model, X_train, y_train, predict, learning_rate):
    print("DeepLearning.fit ---------------")
    print(f"\tModel:\t\t {model}")
    print(f"\tX_train:\t {X_train}")
    print(f"\ty_train:\t {y_train}")
    print(f"\tPredict:\t {predict}")
    print(f"\tLearning rate:\t {learning_rate}")