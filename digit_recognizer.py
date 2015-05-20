import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors


df = pd.read_csv("data.csv", header=0)

y = df["label"]
X = df.drop("label", axis=1)

y = np.array(y)
X = np.array(X)

y_train = y[0:999]
X_train = X[0:999]

y_test = y[1000:1200]
X_test = X[1000:1200]

classifier = neighbors.KNeighborsClassifier()

print 'about to fit'
classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)
print 'score: ', score

prediction = classifier.predict(X[-1])
print prediction


reshaped_example = X[-1].reshape(28,28)
image = plt.imshow(reshaped_example,cmap = cm.Greys_r)
plt.show()