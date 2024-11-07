import numpy as np
import torch

from utils.functions import normalize

# Bu fonksiyon voxel verisini modele vererek beyin görüntüsünü parçalara ayırır
def parcellate(voxel, model, device, mode):
    # Mod'a göre boyutlar ayarlanıyor (coronal, sagittal, axial)
    if mode == "c":
        stack = (224, 192, 192)
    elif mode == "s":
        stack = (192, 224, 192)
    elif mode == "a":
        stack = (192, 224, 192)
    
    # Modeli değerlendirme moduna alıyoruz
    model.eval()
    
    # Voxel verisinin etrafına dolgu ekliyoruz (her eksenin başı ve sonuna)
    voxel = np.pad(
        voxel, [(1, 1), (0, 0), (0, 0)], "constant", constant_values=voxel.min()
    )
    
    # Torch ile inference modunda çalışıyoruz, yani sadece ileri yönlü hesaplama yapılıyor (geri yayılım yok)
    with torch.inference_mode():
        # Çıktı verisi için kutu oluşturuluyor, ilk boyut "stack" değişkenine göre ayarlanıyor
        box = torch.zeros(stack[0], 142, stack[1], stack[2])
        
        # Her bir slice (dilim) üzerinden geçerek model tahminleri yapılıyor
        for i in range(stack[0]):
            i += 1  # Dilim numarası
            
            # Üç boyutlu bir dilim oluşturuluyor, önceki, şimdiki ve sonraki slice'lar bir araya getiriliyor
            image = np.stack([voxel[i - 1], voxel[i], voxel[i + 1]])
            
            # Girdiyi uygun boyuta getirip tensöre dönüştürüyoruz
            image = torch.tensor(image.reshape(1, 3, stack[1], stack[2]))
            
            # Veriyi GPU'ya taşıyoruz (eğer GPU kullanılıyorsa)
            image = image.to(device)
            
            # Model çıktısını hesaplıyoruz ve softmax fonksiyonuyla normalleştiriyoruz
            x_out = torch.softmax(model(image), 1).detach().cpu()
            
            # İlk dilim için kutuya yerleştiriliyor, diğer dilimler sırayla ekleniyor
            if i == 1:
                box[0] = x_out
            else:
                box[i - 1] = x_out
        
        # Kutuyu orijinal boyutlarına geri döndürüp döndürüyoruz
        return box.reshape(stack[0], 142, stack[1], stack[2])

# Parçalama işleminin ana fonksiyonu
def parcellation(voxel, pnet_c, pnet_s, pnet_a, device):
    # Voxel verisini normalize ediyoruz
    voxel = normalize(voxel)

    # Veriyi coronal (ön-arka), sagittal (yan), axial (üst-alt) görünümlerine göre döndürüyoruz
    coronal = voxel.transpose(1, 2, 0)  # Coronal görünüm
    sagittal = voxel  # Sagittal görünüm
    axial = voxel.transpose(2, 1, 0)  # Axial görünüm
    
    # Coronal görünüm için parçalama yapılıyor ve çıktı düzenleniyor
    out_c = parcellate(coronal, pnet_c, device, "c").permute(1, 3, 0, 2)
    
    # Belleği temizliyoruz
    torch.cuda.empty_cache()
    
    # Sagittal görünüm için parçalama yapılıyor ve çıktı düzenleniyor
    out_s = parcellate(sagittal, pnet_s, device, "s").permute(1, 0, 2, 3)
    
    # Belleği tekrar temizliyoruz
    torch.cuda.empty_cache()
    
    # Coronal ve sagittal sonuçlar toplanıyor
    out_e = out_c + out_s
    
    # Belleği boşaltmak için coronal ve sagittal sonuçları siliyoruz
    del out_c, out_s
    
    # Axial görünüm için parçalama yapılıyor ve çıktı düzenleniyor
    out_a = parcellate(axial, pnet_a, device, "a").permute(1, 3, 2, 0)
    
    # Belleği tekrar temizliyoruz
    torch.cuda.empty_cache()
    
    # Axial sonuçlar ekleniyor
    out_e = out_e + out_a
    
    # Axial sonuçlar belleği boşaltmak için siliniyor
    del out_a
    
    # Sonuçların argmax'ı alınıyor (en yüksek olasılıklı sınıf) ve numpy dizisine dönüştürülüyor
    parcellated = torch.argmax(out_e, 0).numpy()
    
    # Parçalanmış sonuç döndürülüyor
    return parcellated
