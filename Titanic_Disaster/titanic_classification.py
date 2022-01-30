import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


#önce veri ön işlemlerini yapıyoruz
def preprocessing(data):
    data.Cabin.fillna('0',inplace=True) #boşlukları doldurur
    #kabin isimlerini sayısal değerlerle değiştirelim
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    
    #cinsiyeti sayısallaştıralım
    data['Sex'].replace('female',1,inplace=True)
    data['Sex'].replace('male',2,inplace=True)
    
    #hangi şehirden binildiğinin bilgisini sayısal değerlerle değiştirelim
    data['Embarked'].replace('S',1,inplace=True)
    data['Embarked'].replace('C',2,inplace=True)
    data['Embarked'].replace('Q',3,inplace=True)
    
    #yaşı boş olanlara ortalama yaşı atayalım
    data['Age'].fillna(data['Age'].mean(),inplace=True)
    
    #ödenen ücreti boş olanlara ortalama ücreti atayalım
    data['Fare'].fillna(data['Fare'].mean(),inplace=True)
    
    #şehir bilgisi boş olanlara medyan olan şehri atayalım
    data['Embarked'].fillna(data['Embarked'].median(),inplace=True)
    
    #boş olan veri olduğunda o satırları tamamen çıkarmak için aşağısı yapılabilir
    #data.dropna(subset=['Fare','Embarked','Age'],inplace=True,how='any')
    
    return data

#veri içerisinde geçen kelimelere göre ayrıştırma yapıp sayısal değer veriyoruz
def group_titles(data):
    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))
    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))
    data['Title'].replace('Master.', 0, inplace=True)
    data['Title'].replace('Mr.', 1, inplace=True)
    data['Title'].replace(['Ms.','Mlle.', 'Miss.'], 2, inplace=True)
    data['Title'].replace(['Mme.', 'Mrs.'], 3, inplace=True)
    data['Title'].replace(['Dona.', 'Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'the'], 4, inplace=True)
    
def data_subset(data):
    #özellikleri çekiyoruz
    features = ['Pclass','SibSp','Parch','Sex','Names','Title','Age','Cabin']
    lenght_features = len(features)
    subset = data[features]
    return subset, lenght_features

#modeli oluşturalım
def create_model(train_set_size,input_length,num_epochs,batch_size):
    model = Sequential()
    #denseleri ekliyoruz
    model.add(Dense(7,input_dim=input_length,activation='softplus'))
    model.add(Dense(3,activation='softplus'))
    model.add(Dense(1,activation='softplus'))
    
    learning_rate = 0.001
    #optimizer seçiyoruz
    adam = Adam(lr=learning_rate)
    
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    #checkpoint kullanarak öğrenilen ağırlıkların en iyisini kaydedelim
    filepath = 'best_weight.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    
    history_model = model.fit(X_train[:train_set_size],Y_train[:train_set_size],callbacks=checkpoint,epochs=num_epochs,batch_size=batch_size,verbose=0)
    return model,history_model

#grafikleri fonksiyonda çizderelim
def plot(history):
    loss_history = history.history["loss"]
    acc_history = history.history["accuracy"]
    
    epochs = [(i+1) for i in range(epoch)]
    ax = plt.subplot(211)
    #211 -> nrows=2, ncols=1, plot_number=1  demek
    ax.plot(epochs,loss_history,color='red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss per epoch')
    
    ax2 = plt.subplot(212)
    ax2.plot(epochs,acc_history,color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy per epoch')
    
    plt.subplot_adjust(hspace=0.8)
    plt.savefig('accuracy_loss.png')
    plt.close()
    
#test işlemleri
def test(batch_size):
    test = pd.read_csv('test.csv',header=0)
    test_ids = test['PassengerId']
    group_titles(test)
    testdata, _ = data_subset(test)
    
    X_test = np.array(testdata).astype(float)
    output = model.predict(X_test,batch_size=batch_size,verbose=0)
    output = output.reshape((418,))
    
    #kişilerin id ve kurulma durumlarını birleşitiriyoruz
    conc_col1 =  np.concatenate((['PassengerId'], test_ids), axis=0)
    conc_col2 =  np.concatenate((['Survived'], output), axis=0)
    
    f = open("output.csv", "w")
    writer = csv.writer(f)
    for i in range(len(conc_col1)):
        writer.writerow([conc_col1[i]] + [conc_col2[i]])
        f.close()
        
seed = 7
np.random.seed(seed)

train = pd.read_csv('train.csv', header=0)
preprocessing(train)
group_titles(train)

epoch = 100
batch_size = 32

traindata, lenght_features = data_subset(train)

Y_train = np.array(train['Survived']).astype(int)
X_train = np.array(traindata).astype(float)

# eğitim ve doğrulama kümeleri oranı
train_set_size = int (.67 * len(X_train))

model, history_model = create_model(train_set_size, lenght_features, epoch, batch_size)

plot(history_model)

X_validation = X_train[train_set_size:]
Y_validation = Y_train[train_set_size:]

loss_and_metrics = model.evaluate(X_validation,Y_validation,batch_size=batch_size)
print("Loss and Metrics")
test(batch_size)