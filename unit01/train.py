import model
import dataset

if __name__ == "__main__":
  algo = "fake-algo"
  model_type = "fake-type"
  x = "all_input_datas"
  y = "all_labels"
  predict = "fake-predict"
  learning_rate = 0.8
  epoch = 100

  dataset = dataset.Dataset(x, y)
  x_train, x_test, y_train, y_test = dataset.split(0.8)

  ai = model.AI(algo, model_type)
  ai.fit(model, x_train, y_train, predict)
  ai.predict(model, x_train)

  dl = model.DeepLearning(algo, model_type)
  dl.train_on_epoch(model, x_train, y_train, epoch, learning_rate)
  dl.fit(model, x_train, y_train, predict, learning_rate)
  dl.predict(model, x_train)

  