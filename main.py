import numpy as np
import matplotlib.pyplot as plt

X = np.load('images_et4/data/trn_img.npy')
Y = np.load('images_et4/data/trn_lbl.npy')


classe=[]
moyenne=[]
for i in range (9):
    classe.append(X[Y==i])
    moyenne.append(np.mean(classe[i], axis=0))

plan=[]
#hyperplans séparateurs
for i in range(9):
    for j in range(i,9):
        vect= np.transpose(moyenne[j]-moyenne[i])
        cste = vect*(-(moyenne[i]+moyenne[j])/2)
        plan.append([vect,cste])

print(plan[0])

dev = np.load('images_et4/data/dev_img.npy')


#afficher une image
img = plan[3][1].reshape(28,28)
plt.imshow(img, plt.cm.gray)
plt.show()


