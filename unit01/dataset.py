class Dataset:
  def __init__(self, input_datas, labels):
    self.input_datas = input_datas
    self.labels = labels
  
  def split(self, percentage):
    print("Dataset.split ---------------")
    print(f"\tPercentage:\t {percentage}")
    x_train = "x-train"
    x_test = "x-test"
    y_train = "y-train"
    y_test = "y-test"
    return x_train, x_test, y_train, y_test
  
  