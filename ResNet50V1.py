from google.colab import drive
drive.mount('/content/drive')

!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

!tar -xf images.tar.gz
!tar -xf annotations.tar.gz

!cp -r images/ /content/drive/MyDrive/petdataset/
!cp -r annotations/ /content/drive/MyDrive/petdataset/

import os

# Kaç görüntü var?
print("Görüntü sayısı:", len(os.listdir("images")))

# Annotations klasörü doğru mu?
print("Annotations içeriği:", os.listdir("annotations"))

import pandas as pd

df = pd.read_csv(
    "annotations/list.txt",
    sep=" ",
    skiprows=6,
    header=None,
    names=["filename", "class_id", "species", "breed_id"]
)

# Dosya yolunu ekle
df["filepath"] = "images/" + df["filename"] + ".jpg"

# Irk ismini dosya adından çek
df["breed_name"] = df["filename"].apply(lambda x: "_".join(x.split("_")[:-1]))

print("Toplam görüntü sayısı:", len(df))
print("Sınıf sayısı:", df["class_id"].nunique())
print("\nİlk 5 satır:")
print(df.head())

import matplotlib.pyplot as plt

# Her ırktan kaç görüntü var
breed_counts = df.groupby("breed_name")["filename"].count().sort_values()

plt.figure(figsize=(15, 8))
breed_counts.plot(kind="barh", color="steelblue")
plt.title("Her Irktan Kaç Görüntü Var?", fontsize=16)
plt.xlabel("Görüntü Sayısı")
plt.ylabel("Irk")
plt.tight_layout()
plt.show()

print("En az görüntü:", breed_counts.min(), "—", breed_counts.idxmin())
print("En fazla görüntü:", breed_counts.max(), "—", breed_counts.idxmax())
print("Ortalama:", round(breed_counts.mean(), 1))

from sklearn.model_selection import train_test_split

# Önce %70 train, %30 geçici
X_train, X_temp, y_train, y_temp = train_test_split(
    df["filepath"], df["class_id"],
    test_size=0.30,
    stratify=df["class_id"],
    random_state=42
)

# %30'u ikiye böl → %15 val, %15 test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("Train seti:     ", len(X_train), "görüntü")
print("Validation seti:", len(X_val), "görüntü")
print("Test seti:      ", len(X_test), "görüntü")
print("Toplam:         ", len(X_train) + len(X_val) + len(X_test))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Train için — augmentation var
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Validation ve Test için — sadece normalize
val_test_datagen = ImageDataGenerator(rescale=1./255)

print("Pipeline hazır ✓")

# DataFrame'leri oluştur
train_df = pd.DataFrame({"filepath": X_train, "class_id": y_train.astype(str)})
val_df   = pd.DataFrame({"filepath": X_val,   "class_id": y_val.astype(str)})
test_df  = pd.DataFrame({"filepath": X_test,  "class_id": y_test.astype(str)})

# Generator'ları bağla
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="class_id",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="filepath",
    y_col="class_id",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_generator = val_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="class_id",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("Train generator:     ", train_generator.samples, "görüntü")
print("Validation generator:", val_generator.samples, "görüntü")
print("Test generator:      ", test_generator.samples, "görüntü")


import numpy as np

# Train generator'dan bir batch al
sample_batch = next(train_generator)
images = sample_batch[0]  # görüntüler

# 9 tanesini göster
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.axis("off")

plt.suptitle("Augmentation Uygulanmış Örnek Görüntüler", fontsize=14)
plt.tight_layout()
plt.show()


import json

# Sınıf isimlerini kaydet — Gökay kullanacak
class_names = dict(zip(
    train_generator.class_indices.values(),
    train_generator.class_indices.keys()
))

with open("/content/drive/MyDrive/petdataset/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Kaydedilen dosyalar:")
print("✅ class_names.json → Gökay için")
print("✅ train_generator  → 5144 görüntü, 37 sınıf")
print("✅ val_generator    → 1102 görüntü, 37 sınıf")
print("✅ test_generator   → 1103 görüntü, 37 sınıf")


import os

path = "/content/drive/MyDrive/petdataset"

if os.path.exists(path):
    print("Klasör bulundu! İçeriği:")
    print(os.listdir(path))
else:
    print("Klasör bulunamadı!")
