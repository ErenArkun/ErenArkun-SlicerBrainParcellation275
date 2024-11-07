import numpy as np
import torch
from scipy.ndimage import binary_closing

from utils.functions import normalize


def crop(voxel, model, device):
    model.eval()  # Modeli değerlendirme moduna al
    with torch.inference_mode():  # İnference (tahmin) modunda çalış, gereksiz hesaplamaları devre dışı bırak
        output = torch.zeros(256, 256, 256).to(device)  # Çıkış için boş bir tensor oluştur
        for i, v in enumerate(voxel):  # Voxel'deki her bir parça için döngü başlat
            image = v.reshape(1, 1, 256, 256)  # Voxel parçasını uygun şekle getir
            image = torch.tensor(image).to(device)  # Tensor'u cihaza (CPU/GPU) taşımak için tensor oluştur
            x_out = torch.sigmoid(model(image)).detach()  # Modelden tahmin al, sigmoid aktivasyon fonksiyonunu uygula
            if i == 0:  # İlk parça için
                output[0] = x_out  # İlk tahmini çıkış tensor'üne ekle
            else:  # Diğer parçalar için
                output[i] = x_out  # Tahmini çıkış tensor'üne ekle
        return output.reshape(256, 256, 256)  # Çıkış tensor'unu 256x256x256 boyutuna yeniden şekillendir


def closing(voxel):
    selem = np.ones((3, 3, 3), dtype="bool")  # 3x3x3 boyutunda bir yapısal eleman oluştur
    voxel = binary_closing(voxel, structure=selem, iterations=3)  # Binary kapanma işlemi uygula
    return voxel  # İşlenmiş voxel'i döndür


def cropping(data, cnet, device):
    voxel = data.get_fdata()  # Veriden 3D voxel verisini al
    voxel = normalize(voxel)  # Voxel verisini normalize et

    coronal = voxel.transpose(1, 2, 0)  # Voxel verisini koronal düzlemde döndür
    sagittal = voxel  # Sagital düzlem için voxel verisini al
    out_c = crop(coronal, cnet, device).permute(2, 0, 1)  # Koronal düzlemde cropping yap ve boyutları değiştir
    out_s = crop(sagittal, cnet, device)  # Sagital düzlemde cropping yap
    out_e = ((out_c + out_s) / 2) > 0.5  # Koronal ve sagital çıktıları birleştir ve eşik uygula
    out_e = out_e.cpu().numpy()  # Çıktıyı NumPy dizisine dönüştür
    out_e = closing(out_e)  # Kapanma işlemini uygula
    cropped = data.get_fdata() * out_e  # Orijinal veriyi, kapanma sonrası elde edilen maske ile çarp
    return cropped  # Kırpılmış veriyi döndür
