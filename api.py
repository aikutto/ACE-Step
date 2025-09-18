from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from acestep.pipeline_ace_step import ACEStepPipeline
import uuid
import torch

app = FastAPI(title="ACEStep Pipeline API")

# İstemci talebi (request) için Pydantic modeli
class ACEStepInput(BaseModel):
    checkpoint_path: str
    bf16: bool = True
    torch_compile: bool = False
    device_id: int = 0
    output_path: Optional[str] = None
    audio_duration: float
    prompt: str
    lyrics: str
    infer_step: int
    guidance_scale: float
    scheduler_type: str
    cfg_type: str
    omega_scale: float
    actual_seeds: List[int]
    guidance_interval: float
    guidance_interval_decay: float
    min_guidance_scale: float
    use_erg_tag: bool
    use_erg_lyric: bool
    use_erg_diffusion: bool
    oss_steps: List[int]
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0

# Yanıt (response) için Pydantic modeli
class ACEStepOutput(BaseModel):
    status: str
    output_path: Optional[str]
    message: str

# Global değişken olarak pipeline nesnesini tanımlayın
model_pipeline = None
global_params = {}

def initialize_pipeline(checkpoint_path: str, bf16: bool, torch_compile: bool, device_id: int):
    """
    Pipeline nesnesini başlatmak için yardımcı fonksiyon.
    """
    global model_pipeline, global_params
    
    # Parametreler değişmediyse, mevcut pipeline'ı kullan
    current_params = {
        'checkpoint_path': checkpoint_path, 
        'bf16': bf16, 
        'torch_compile': torch_compile, 
        'device_id': device_id
    }
    if model_pipeline and global_params == current_params:
        print("Mevcut pipeline yeniden kullanılıyor.")
        return model_pipeline

    print("Yeni pipeline başlatılıyor...")
    try:
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        # Cihaz belirleme ve dtype ayarlama
        device = f"cuda:{device_id}" if device_id is not None and torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if bf16 and device.startswith("cuda") else torch.float32

        model_pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            dtype=str(dtype).split('.')[-1],
            torch_compile=torch_compile,
            device=device
        )
        
        # Yeni parametreleri kaydet
        global_params = current_params
        return model_pipeline
        
    except Exception as e:
        print(f"Pipeline başlatılırken hata oluştu: {e}")
        raise

# Uygulama başladıktan hemen sonra modeli başlatın
# Cloud Run'ın container'ı başlatıldığında bu kod çalışır
@app.on_event("startup")
async def startup_event():
    # Burada model için varsayılan bir başlangıç parametresi belirleyebilirsiniz.
    # Örneğin, en sık kullanılan checkpoint ve ayarları.
    # Bu, ilk gelen isteğe kadar modeli belleğe yükler.
    # Eğer bu kısmı atlamak isterseniz, modeli ilk istekte yükleyebilirsiniz.
    print("Uygulama başlatılıyor, pipeline yükleniyor...")
    # initialize_pipeline("/app/model_checkpoint", True, False, 0)
    # Bu satırı yorumdan çıkararak uygulamanın başlangıcında modeli yükleyebilirsiniz.
    pass

@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput):
    try:
        # Pipeline'ı ilk istekte yükleyin veya mevcut olanı kullanın
        # Bu yaklaşım, modelin ilk istekte yüklenmesini sağlar ve sonraki istekler için hız kazandırır.
        model_demo = initialize_pipeline(
            input_data.checkpoint_path,
            input_data.bf16,
            input_data.torch_compile,
            input_data.device_id
        )

        # Parametreleri hazırlayın
        params = (
            input_data.audio_duration,
            input_data.prompt,
            input_data.lyrics,
            input_data.infer_step,
            input_data.guidance_scale,
            input_data.scheduler_type,
            input_data.cfg_type,
            input_data.omega_scale,
            ", ".join(map(str, input_data.actual_seeds)),
            input_data.guidance_interval,
            input_data.guidance_interval_decay,
            input_data.min_guidance_scale,
            input_data.use_erg_tag,
            input_data.use_erg_lyric,
            input_data.use_erg_diffusion,
            ", ".join(map(str, input_data.oss_steps)),
            input_data.guidance_scale_text,
            input_data.guidance_scale_lyric,
        )

        # Çıktı yolu oluşturun
        output_path = input_data.output_path or f"output_{uuid.uuid4().hex}.wav"

        # Pipeline'ı çalıştırın
        model_demo(
            *params,
            save_path=output_path
        )

        return ACEStepOutput(
            status="success",
            output_path=output_path,
            message="Audio generated successfully"
        )

    except Exception as e:
        # Hata durumunda HTTP 500 yanıtı döndürün
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")

@app.get("/health")
async def health_check():
    # Sağlık kontrolü
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Bu kısım sadece yerel test için kullanılır, Cloud Run'da Uvicorn doğrudan çağrılır.
    uvicorn.run(app, host="0.0.0.0", port=8000)