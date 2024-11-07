import torch  # PyTorch kütüphanesini içe aktar
from scipy.ndimage import binary_dilation  # İkili genişletme işlemi için SciPy'nin ndimage modülünden binary_dilation fonksiyonunu içe aktar

from utils.functions import normalize  # normalize fonksiyonunu utils.functions modülünden içe aktar

def separate(voxel, model, device, mode):
    # Voxel verisini belirtilen model ile ayırmak için fonksiyon
    if mode == "c":
        stack = (224, 192, 192)  # Koronal kesit için boyutları ayarla
    elif mode == "a":
        stack = (192, 224, 192)  # Aksiyel kesit için boyutları ayarla

    model.eval()  # Modeli değerlendirme moduna al
    with torch.inference_mode():  # İleri besleme sırasında gradyan hesaplamalarını devre dışı bırak
        output = torch.zeros(stack[0], 3, stack[1], stack[2]).to(device)  # Çıktı tensorünü sıfırlarla
        for i, v in enumerate(voxel):  # Voxel üzerindeki her bir dilim için döngü
            image = torch.tensor(v.reshape(1, 1, stack[1], stack[2]))  # Voxel dilimini tensor'e dönüştür
            image = image.to(device)  # Tensor'u belirtilen cihaza (GPU/CPU) taşı
            x_out = torch.softmax(model(image), 1).detach()  # Model çıktısını elde et ve softmax uygula
            if i == 0:  # İlk dilim ise
                output[0] = x_out  # İlk çıktıyı ayarla
            else:
                output[i] = x_out  # Diğer çıktıları ayarla
        return output  # Çıktıyı döndür

def hemisphere(voxel, hnet_c, hnet_a, device):
    # Voxel verisini hemisferlere ayırmak için fonksiyon
    voxel = normalize(voxel)  # Voxel verisini normalize et

    coronal = voxel.transpose(1, 2, 0)  # Koronal kesit için voxel'i döndür
    transverse = voxel.transpose(2, 1, 0)  # Aksiyel kesit için voxel'i döndür
    out_c = separate(coronal, hnet_c, device, "c").permute(1, 3, 0, 2)  # Koronal kesitten çıktıyı al
    out_a = separate(transverse, hnet_a, device, "a").permute(1, 3, 2, 0)  # Aksiyel kesitten çıktıyı al
    out_e = out_c + out_a  # İki çıktı üzerinde toplama işlemi yap
    out_e = torch.argmax(out_e, 0).cpu().numpy()  # En yüksek değere sahip olan indeksi al ve numpy dizisine çevir
    torch.cuda.empty_cache()  # GPU bellek önbelleğini temizle

    # Maskeleri oluştur
    dilated_mask_1 = binary_dilation(out_e == 1, iterations=5).astype("int16")  # İlk maske için genişletme uygula
    dilated_mask_1[out_e == 2] = 2  # İkinci maske için değerleri ayarla
    dilated_mask_2 = (
        binary_dilation(dilated_mask_1 == 2, iterations=5).astype("int16") * 2  # İkinci genişletilmiş maske
    )
    dilated_mask_2[dilated_mask_1 == 1] = 1  # İlk maskenin değerlerini ayarla
    return dilated_mask_2  # Genişletilmiş maskeyi döndür
