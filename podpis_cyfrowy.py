import numpy as np
import cv2  
import streamlink
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA3_256
from Crypto.Signature import pkcs1_15

# Pobranie obrazu z url streama
def getImage(url):
    stream = streamlink.streams(url)
    stream_url = stream["best"].url

    streamCap = cv2.VideoCapture(stream_url)
    success, image = streamCap.read()

    if success:
        print("Image is saved successfully.")
    else:
        print("Image is not saved successfully.")

    streamCap.release()

    image=imageCorrection(image)
    return image

# skalowanie obrazu, zapis obrazu w grayscale
def imageCorrection(image):
    image = cv2.resize(image, (1024,1024))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def binarize(input, c):
    return (input > c).astype(int)

def xor_planes(plane, binary_map):
    for x in range(len(plane)):
        plane[x] = np.bitwise_xor(plane[x], binary_map[x])
    return plane

# Generowanie mapy logistycznej
def generate_logistic_map(la, x0, length, cut):
    logisticMap = np.zeros(length + cut)
    logisticMap[0] = x0
    for i in range(1, length + cut):
        logisticMap[i] = la * logisticMap[i - 1] * (1 - logisticMap[i - 1])
    return logisticMap[cut:]

def chaos_image_mix(sequence, image):
    indices = np.argsort(sequence)
    new_image = np.empty_like(image)
    size = image.shape[0]

    for y in range(size):
        for x in range(size):
            new_image[y, x] = image[indices[(y * size) + x] % size, indices[(y * size) + x] % size]

    cv2.imwrite("C:/Desktop/img1.png", new_image)
    return new_image

# Dzieli wejsciowa tablice bitowa na osiem plaszczyzn bitowych
def split_bit_planes(input):
    a = np.unpackbits(input)
    
    plane_length = a.size // 8
    planes = [np.empty(plane_length, dtype=np.uint8) for _ in range(8)]
    
    for i in range(plane_length):
        for j in range(8):
            planes[j][i] = a[i * 8 + j]
    
    return tuple(planes)

# Laczy osiem plaszczyzn bitowych w jedna tablice bitowa
def merge_planes(*args):
    num_planes = len(args)
    if num_planes != 8:
        raise ValueError("merge_planes requires exactly 8 input arrays.")

    lengths = [len(arr) for arr in args]
    if len(set(lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")

    result_size = lengths[0] * 8
    result = np.empty(result_size, dtype=np.uint8)

    for i in range(lengths[0]):
        for j in range(8):
            result[i * 8 + j] = args[j][i]

    reconstructed = np.packbits(result)
    return reconstructed

def image_processing(image):
    data = np.array(image, dtype=np.uint8)
    size = data.shape

    logistic_maps = [generate_logistic_map(4, 0.361 + i * 0.001, size[0] * size[1], 250) for i in range(9)]
    
    new_image = chaos_image_mix(logistic_maps[0], data)

    binary_maps = [binarize(logistic_maps[i + 1], 0.5) for i in range(8)]

    p1, p2, p3, p4, p5, p6, p7, p8 = split_bit_planes(new_image)

    p1 = xor_planes(p1, binary_maps[0])
    p2 = xor_planes(p2, binary_maps[1])
    p3 = xor_planes(p3, binary_maps[2])
    p4 = xor_planes(p4, binary_maps[3])
    p5 = xor_planes(p5, binary_maps[4])
    p6 = xor_planes(p6, binary_maps[5])
    p7 = xor_planes(p7, binary_maps[6])
    p8 = xor_planes(p8, binary_maps[7])

    res = merge_planes(p1, p2, p3, p4, p5, p6, p7, p8)
    return res

#####################################################################

class RandomBytes:
    def __init__(self, source):
        self.source = source
        self.iter = 0
    
    def read(self, n):
        result = self.source[self.iter:self.iter+n]
        self.iter += n
        return bytes(result)

# Generowanie kluczy
def generate_keys(random):
    key = RSA.generate(2048, randfunc=lambda n: random.read(n))
    private_key = key
    public_key = key.publickey()
    return private_key, public_key

def random_bits(img):
    res = image_processing(img)
    random = RandomBytes(res)
    private_key, public_key = generate_keys(random)
    return private_key, public_key

# Laczenie pobranych obrazow
def mergeImages(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("Images do not have the same sizes.")
    
    block_size=16
    img = img1.copy()
    size = img1.shape[0]

    for y in range(0, size, block_size):
        for x in range(0, size, block_size):
            if ((x+y)//block_size) % 2 == 0:
                img[y:y+block_size, x:x+block_size] = img2[y:y+block_size, x:x+block_size]
    return img

img1 = getImage("https://www.youtube.com/watch?v=tFZLxK0nGJQ")
img2 = getImage("https://www.youtube.com/watch?v=GakUUW9Anpo")
img = mergeImages(img1, img2)

cv2.imwrite("C:/Desktop/img.png", img)

#####################################################################
# Podpis cyfrowy

private_key, public_key = random_bits(img)
with open('C:/Desktop/document.txt', 'r') as file:
    document = file.read()
    document = bytes(document, 'utf-8')

hash_A = SHA3_256.new(document)
podpis = pkcs1_15.new(private_key).sign(hash_A)

print('\n')
print('---Podpis cyfrowy - test---')
print("Dokument:", document)
print("Skrót wiadomości:", hash_A.hexdigest())
print("Podpis:", podpis.hex())
print('\n')

while(True):
    opcja = int(input("Wybierz scenariusz (1 - podpis prawidlowy, 2 - modyfikacja dokumentu, 3 - wyjscie):"))
    if(opcja == 1):   
        document_R = document
        
        # Strona odbiorcza oblicza hash odebranej wiadomosci
        hash_B = SHA3_256.new(document_R)
        print("Skrót otrzymanej wiadomości:", hash_B.hexdigest())

        # Weryfikacja podpisu, porownanie hasha B z hashem A 
        try:
            pkcs1_15.new(public_key).verify(hash_B, podpis)
            print("Podpis cyfrowy poprawny")
        except (ValueError, TypeError):
            print("Podpis cyfrowy nie jest poprawny")

    elif(opcja == 2):
        document_R = b'123'
        # Strona odbiorcza oblicza hash odebranej wiadomosci
        hash_B = SHA3_256.new(document_R)
        print("Skrót otrzymanej wiadomości:", hash_B.hexdigest())
        
        # Weryfikacja podpisu, porownanie hasha B z hashem A 
        try:
            pkcs1_15.new(public_key).verify(hash_B, podpis)
            print("Podpis cyfrowy poprawny")
        except (ValueError, TypeError):
            print("Podpis cyfrowy nie jest poprawny")

    elif(opcja == 3):
        break
    else: 
        print("Nie ma takiego scenariusza.")
    print('\n')

#####################################################################