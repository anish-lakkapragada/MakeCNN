# just to test the API
from Class_project import AutoWork

aw = AutoWork("/Users/anish/Downloads/xray_dataset_covid19")
aw.train(epochs = 1)
aw.predict("/Users/anish/Downloads/xray_dataset_covid19/Testing/normal/NORMAL2-IM-0120-0001.jpeg")
