from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

veri = pd.read_csv("breast-cancer-wisconsin.data") #veriyi bu dosyanda cektik

veri.replace('?', -99999, inplace=True) #veri icnde soru isareti olan yerleri degistirdik
#veri.drop(['id'], axis=1)
veriyeni = veri.drop(['1000025'],axis=1)

imp = Imputer(missing_values=-99999, strategy="mean",axis=0)
veriyeni = imp.fit_transform(veriyeni)


giris = veriyeni[:,0:8]
cikis = veriyeni[:,9]                   

model = Sequential()
model.add(Dense(256,input_dim=8)) #ilk hidden layerimiz 256 hucreye sahip
model.add(Activation('relu'))
model.add(Dense(256))   #ikinci hidden layerimiz 256 hucreye sahip
model.add(Activation('relu'))
model.add(Dense(256)) #ucuncu hidden layerimiz 256 hucreye sahip
model.add(Activation('softmax')) # son hidden layera softmax eklemeliyiz

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(giris,cikis,epochs=50,batch_size=32,validation_split=0.10) #validation split olarak ayırdık. Yani algoritmamız X’in %90'ını 
# kullanarak modelimizi eğitecek, geri kalan yüzde 10 ile test edecek ve hatalarını düzeltecek.


tahmin = np.array([5,5,5,8,10,8,7,3]).reshape(1,8)  #burda datadan aldigmiz bir verinin 2 veya 4 olacagini bulmaya calsitik
print(model.predict_classes(tahmin))
