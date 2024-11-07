import pickle  # Python nesnelerini dosyadan okuma ve yazma işlemleri için pickle kütüphanesini içe aktar
import numpy as np  # NumPy kütüphanesini içe aktar (sayısal işlemler için)
import torch  # PyTorch kütüphanesini içe aktar (derin öğrenme için)
import os  # İşletim sistemi ile etkileşim için os kütüphanesini içe aktar

def postprocessing(parcellated, separated, shift, device):
    # Post-processing (işlem sonrası) fonksiyonu tanımla. Giriş olarak parcellated, separated, shift ve device alır.

    script_dir = os.path.dirname(os.path.realpath(__file__))  # Mevcut script'in bulunduğu dizinin yolunu al
    
    with open(os.path.join(script_dir, "split_map.pkl"), "rb") as tf:
        # 'split_map.pkl' dosyasını aç ve içeriğini yükle
        dictionary = pickle.load(tf)  # Yüklenen veriyi dictionary (sözlük) olarak ata
    pmap = torch.tensor(parcellated.astype("int16"), requires_grad=False).to(device)  # Parcellated verisini PyTorch tensor'una çevir, grad hesaplamasını devre dışı bırak ve cihaza taşı
    hmap = torch.tensor(separated.astype("int16"), requires_grad=False).to(device)  # Separated verisini aynı şekilde PyTorch tensor'una çevir ve cihaza taşı
    combined = torch.stack((torch.flatten(hmap), torch.flatten(pmap)), axis=-1)  # hmap ve pmap tensor'lerini düzleştir ve birleştir
    output = torch.zeros_like(hmap).ravel()  # hmap ile aynı boyutta sıfırlardan oluşan bir tensor oluştur ve düzleştir

    for key, value in dictionary.items():  # Dictionary'deki her bir anahtar-değer çifti için döngü başlat
        key = torch.tensor(key, requires_grad=False).to(device)  # Anahtarı PyTorch tensor'una çevir ve cihaza taşı
        mask = torch.all(combined == key, axis=1)  # Anahtar ile eşleşen öğeleri bulmak için maske oluştur
        output[mask] = value  # Maske ile belirlenen öğeleri değeri ile güncelle

    output = output.reshape(hmap.shape)  # Çıktıyı orijinal hmap boyutuna yeniden şekillendir
    output = output.cpu().detach().numpy()  # Çıktıyı CPU'ya taşı, gradyan hesaplamasını kaldır ve NumPy dizisine çevir
    output = output * (  # Çıktıyı bir maske ile çarp
        np.logical_or(  # İki koşulu birleştirerek bir maske oluştur
            np.logical_or(separated > 0, parcellated == 87),  # Separated > 0 veya parcellated 87'ye eşitse
            parcellated == 138  # Ya da parcellated 138'e eşitse
        )
    )
    output = np.pad(  # Çıktıya kenar boşluğu ekle
        output, [(32, 32), (16, 16), (32, 32)], "constant", constant_values=0  # Kenar boşluğu için boyutları ve sabit değeri belirt
    )
    output = np.roll(output, (-shift[0], -shift[1], -shift[2]), axis=(0, 1, 2))  # Çıktıyı belirtilen kaydırmalarla kaydır (roll)
    return output  # Sonucu döndür
