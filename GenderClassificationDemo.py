from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#clf = tree.DecisionTreeClassifier()
#clf = KNeighborsClassifier()
#clf = LogisticRegression()
#clf = GaussianNB()
clf = RandomForestClassifier()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43],
     [160, 60, 38], [154, 54, 37],
     [166, 65, 40],[190, 90, 47],
     [175, 64, 39],[177, 70, 40],
     [159, 55, 37], [171, 75, 42],
     [181, 85, 43]]
y = ['male', 'male',
     'female', 'female',
     'male', 'male',
     'female', 'female',
     'female', 'male',
     'male']

#Train them on data
clf = clf.fit(X,y)

prediction = clf.predict([[182,81,45]])

print(prediction[0])