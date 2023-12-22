import torch
import torchvision

# 加载预训练模型
model = torchvision.models.resnet50(pretrained = True)
model.eval()

# 创建一个dummy输入，与你的模型输入相匹配
dummy_input = torch.randn(1,3,224,224)

# 导出模型
torch.onnx.export(model, dummy_input, "resnet50.onnx", export_params=True, opset_version=11)
