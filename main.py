import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

X = np.load('images_et4/data/trn_img.npy')
Y = np.load('images_et4/data/trn_lbl.npy')

classe=[]
moyenne=[]

#barycentres
for i in range(10):
    classe.append(X[Y == i])
    moyenne.append(np.mean(classe[i], axis=0))

dev = np.load('images_et4/data/dev_img.npy')
devLabel = np.load('images_et4/data/dev_lbl.npy')

result = []

for j in range(5000):
    minimumDistance = 10000000
    minimumIndex = 0
    for i in range(10):
        distance = np.linalg.norm(dev[j] - moyenne[i])
        if distance < minimumDistance:
            minimumIndex = i
            minimumDistance = distance
    result.append(minimumIndex)

finalResult = result != devLabel
print("Taux d'images non reconnues : ")
# Most beautiful thing ever
print(finalResult[finalResult].size/finalResult.size)

'''values = [10, 20, 50, 100, 250, 300]
for k in values:
    pca = PCA(n_components=k)
    pca.fit(X)
    reducedX = pca.transform(X)

    classe=[]
    moyenne=[]
    #barycentres
    for i in range(10):
        classe.append(reducedX[Y == i])
        moyenne.append(np.mean(classe[i], axis=0))

    result = []
    reducedDev = pca.transform(dev)

    for j in range(5000):
        minimumDistance = 10000000
        minimumIndex = 0
        for i in range(10):
            distance = np.linalg.norm(reducedDev[j] - moyenne[i])
            if distance < minimumDistance:
                minimumIndex = i
                minimumDistance = distance
        result.append(minimumIndex)

    finalResult = result != devLabel
    print("Taux d'images non reconnues : ")
    # Most beautiful thing ever
    print(finalResult[finalResult].size/finalResult.size, "with nb components = ", k)
'''
pca = PCA(n_components=50)
newX = pca.fit_transform(X)
clf = LinearSVC(random_state=0, tol=0.1)
'''
print(clf.get_params())
clf.fit(newX, Y)

newDev = pca.transform(dev)

svmResult = clf.predict(newDev)

finalResult = svmResult != devLabel

print("Taux d'images non reconnues par le svm : ")
# Most beautiful thing ever
print(finalResult[finalResult].size/finalResult.size)
'''



neighbors = KNeighborsClassifier(n_neighbors=10)
print("pouloulou")
neighbors.fit(X, Y)
print("pouloulou")

neighborsResult = neighbors.predict(dev)
print("pouloulou")
finalResult = neighborsResult != devLabel

print("Taux d'images non reconnues par le k-neighbors : ")
# Most beautiful thing ever
print(finalResult[finalResult].size/finalResult.size)