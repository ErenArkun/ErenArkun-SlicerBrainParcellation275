import os  # İşletim sistemi ile etkileşim kurmak için os modülünü içe aktar

import nibabel as nib  # NIfTI dosyalarını okumak ve yazmak için nibabel kütüphanesini içe aktar
import SimpleITK as sitk  # Görüntü işleme için SimpleITK kütüphanesini içe aktar
from nibabel import processing  # Nibabel'in işleme modülünü içe aktar
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform  # Nibabel'den dönüşüm işlevlerini içe aktar


def N4_Bias_Field_Correction(input_path, output_path):
    # N4 Bias Alanı Düzeltme fonksiyonu. Giriş ve çıkış dosya yollarını alır.
    
    raw_img_sitk = sitk.ReadImage(input_path, sitk.sitkFloat32)  # Giriş dosyasını SimpleITK ile oku ve float32 formatına dönüştür
    transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)  # Görüntü yoğunluğunu 0-255 aralığına yeniden ölçeklendir
    transformed = sitk.LiThreshold(transformed, 0, 1)  # Görüntüyü Li eşikleme yöntemi ile ikili hale getir
    head_mask = transformed  # Elde edilen ikili görüntüyü baş maskesi olarak ayarla
    shrinkFactor = 4  # Boyut küçültme faktörünü belirle
    inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())  # Giriş görüntüsünü küçült
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())  # Maske görüntüsünü küçült
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()  # N4 düzeltici filtreyi oluştur
    corrected = bias_corrector.Execute(inputImage, maskImage)  # Düzeltme işlemini uygula
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)  # Logaritmik bias alanını al
    corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)  # Düzeltme işlemini orijinal çözünürlükte uygula
    sitk.WriteImage(corrected_image_full_resolution, output_path)  # Düzeltme sonucu görüntüyü dosyaya yaz
    return  # Fonksiyonu sonlandır


def preprocessing(ipath, save):
    # Ön işleme fonksiyonu. Giriş dosyası yolunu ve kaydedilecek dosya adını alır.
    
    opath = f"N4/{save}.nii"  # Çıktı dosya yolunu oluştur
    os.makedirs("N4", exist_ok=True)  # "N4" klasörünü oluştur, varsa hata verme
    N4_Bias_Field_Correction(ipath, opath)  # N4 düzeltme fonksiyonunu çağır
    odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(opath)))  # Çıktı görüntüsünü yükle ve en yakın kanonik forma getir
    data = processing.conform(  # Görüntüyü belirli bir şekle ve voxel boyutuna uydur
        odata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1  # Hedef şekil ve voxel boyutunu ayarla
    )
    return odata, data  # İşlemden elde edilen orijinal ve işlenmiş verileri döndür
