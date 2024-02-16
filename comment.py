import torch  # PyTorch kütüphanesini içeri aktarır.
from PIL import (
    Image,
)  # Görüntü dosyalarını işlemek için PIL kütüphanesinden Image modülünü içeri aktarır.
from misc import colorize  # colorize fonksiyonunu içeri aktarır.


class DepthEstimationModel:
    def __init__(self) -> None:
        self.device = self._get_device()  # Cihazı (CUDA veya CPU) belirler.
        print(self.device)
        self.model = self._initialize_model(  # Derinlik tahmini modelini yükler ve etkinleştirir.
            model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"
        ).to(
            self.device
        )

    def _get_device(self):
        return (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # CUDA kullanılabilirse CUDA'yı, aksi takdirde CPU'yu döndürür.

    def _initialize_model(self, model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"):
        torch.hub.help(
            "intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True
        )  # Model hakkında bilgi alır.
        model = (
            torch.hub.load(  # Önceden eğitilmiş bir derinlik tahmini modelini yükler.
                model_repo, model_name, pretrained=True, skip_validation=False
            )
        )
        model.eval()  # Modeli değerlendirme moduna alır.
        print("Model initialized.")  # Modelin başarıyla başlatıldığını bildirir.
        return model

    def save_colored_depth(self, depth_numpy, output_path):
        colored = colorize(
            depth_numpy
        )  # Siyah-beyaz derinlik haritasını renkliye dönüştürür.
        Image.fromarray(colored).save(
            output_path
        )  # Renkli derinlik haritasını kaydeder.
        print("Image saved.")  # Görüntünün başarıyla kaydedildiğini bildirir.

    def calculate_depthmap(self, image_path, output_path):
        image = Image.open(image_path).convert(
            "RGB"
        )  # Görüntüyü açar ve RGB formatına dönüştürür.
        print("Image read.")  # Görüntünün başarıyla okunduğunu bildirir.
        depth_numpy = self.model.infer_pil(
            image
        )  # Görüntüden derinlik haritasını hesaplar.
        self.save_colored_depth(
            depth_numpy, output_path
        )  # Renkli derinlik haritasını kaydeder.
        return f"Image saved to {output_path}"  # Kaydedilen görüntünün yolunu döndürür.


model = DepthEstimationModel()  # Derinlik tahmini modelini oluşturur.
model.calculate_depthmap(
    "./test_image.png", "output_image.png"
)  # Verilen bir test görüntüsünden derinlik haritası hesaplar ve çıktı görüntü dosyasına kaydeder.
