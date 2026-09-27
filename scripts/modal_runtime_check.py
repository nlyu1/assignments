"""Check the CUDA/compiler runtime before long training runs."""
import modal
from modal_utils import build_image
app = modal.App('cs312-runtime-check')
@app.function(image=build_image(), gpu='H100', timeout=180)
def check():
    import shutil
    import torch
    print('GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, 'gcc:', shutil.which('gcc'))
    from triton.compiler.compiler import triton_key
    print('Triton import OK')
    x = torch.randn(1024, device='cuda')
    y = torch.compile(lambda a: a.sin() + a.cos())(x)
    torch.cuda.synchronize()
    print('Compiled CUDA kernel OK:', torch.isfinite(y).all().item())
@app.local_entrypoint()
def main():
    check.remote()
