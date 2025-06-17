import torch
from ultralytics.nn.modules.head import MultiTaskDetect

def test_multitask_detect_forward():
    ch = (64, 128, 256)
    head = MultiTaskDetect(nc_list=[2, 3], ch=ch)
    sizes = [80, 40, 20]
    x = [torch.randn(1, c, sizes[i], sizes[i]) for i, c in enumerate(ch)]
    head.eval()
    outputs = head(x)
    assert isinstance(outputs, list)
    assert len(outputs) == 2

