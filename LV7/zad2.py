
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1, 7):
    if i == 4:
        continue 

    img = Image.imread(f"imgs\\test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title(f"Originalna slika test_{i}.jpg")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255

    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    labels = km.predict(img_array_aprox)

    centroids = km.cluster_centers_

    img_array_aprox[:, 0] = centroids[labels][:, 0]
    img_array_aprox[:, 1] = centroids[labels][:, 1]
    img_array_aprox[:, 2] = centroids[labels][:, 2]
    img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img)
    axarr[1].imshow(img_array_aprox)
    plt.tight_layout()
    plt.show()

K = 5 
labels_reshaped = np.reshape(labels, (w, h))

for k in range(K):
    # Kreiraj praznu binarnu sliku (svi pikseli crni)
    binary_img = np.zeros((w, h), dtype=np.uint8)

    # Postavi piksele koji pripadaju klasteru k na bijelo (255)
    binary_img[labels_reshaped == k] = 255

    # Prikaz binarne slike
    plt.figure()
    plt.title(f"Binarna slika za grupu {k}")
    plt.imshow(binary_img, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # ucitaj sliku
img = Image.imread("imgs\\test_4.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# # pretvori vrijednosti elemenata slike u raspon 0 do 1
# img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

km = KMeans(n_clusters=5, init="k-means++", n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

centroids = km.cluster_centers_

img_array_aprox[:, 0] = centroids[labels][:, 0]
img_array_aprox[:, 1] = centroids[labels][:, 1]
img_array_aprox[:, 2] = centroids[labels][:, 2]
img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

f, axarr = plt.subplots(1, 2)

# Prikaz originalne slike
axarr[0].imshow(img)
axarr[0].set_title("Originalna slika")

# Prikaz aproksimirane slike (sa 5 boja)
axarr[1].imshow(img_array_aprox)
axarr[1].set_title("Kvantizirana slika (5 boja)")

plt.tight_layout()
plt.show()

inertias = []
K_values = range(1, 11)

for k in K_values:
    km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=0)
    km.fit(img_array)  # koristi se originalni niz piksela
    inertias.append(km.inertia_)

# Plot
plt.figure()
plt.plot(K_values, inertias, marker='o')
plt.xlabel("Broj klastera K")
plt.ylabel("Inercija (J)")
plt.title("Elbow metoda – ovisnost J o broju grupa K")
plt.grid(True)
plt.tight_layout()
plt.show()
