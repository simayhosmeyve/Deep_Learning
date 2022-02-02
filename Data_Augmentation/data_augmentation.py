from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#Kullanılacak veri arttırma tekniklerini tanımlayalım

datagen = ImageDataGenerator(rotation_range=40, 
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')
#rotation_range = döndürme oranı
#width_shift_range = genişlik değiştirme
#height_shift_range = yükseklik değiştirme
#shear_range = görüntünün belirli bir kısmını kesme
#zoom_range = yakınlaştırma oranını değiştirme

#Arttıracağımız görüntüleri yükleyelim
img = load_img(r'image_example\3.jpg')
x = img_to_array(img)
x = x.reshape((1,)+x.shape)

#Arttırma işlemi
#Önceden tanımladığımız yöntemlerden rastgele seçilerek uygulanacak
#5er adet üretelim
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='image_augmentation_example',save_format='jpg'):
    i+=1
    if i>5:
        break