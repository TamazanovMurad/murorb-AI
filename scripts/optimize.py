import torch.quantization
from model import VoiceConversionModel

# Lade das Modell
model = VoiceConversionModel()
model.load_state_dict(torch.load("outputs/trained_model.pth"))
model.eval()

# Quantisiere das Modell (8-Bit-Precision)
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

# Speichere optimiertes Modell
torch.jit.save(torch.jit.script(model), "outputs/optimized_model.pt")