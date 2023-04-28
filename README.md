---
language:
- zh
- en
tags:
- chatglm
- glm
- onnx
- onnxruntime
---

# ChatGLM-6B + ONNX

This model is exported from [ChatGLM-6b](https://huggingface.co/THUDM/chatglm-6b) with int8 quantization and optimized for [ONNXRuntime](https://onnxruntime.ai/) inference.

Inference code with ONNXRuntime is uploaded with the model. Install requirements and run `streamlit run web-ui.py` to start chatting. Currently the `MatMulInteger` (for u8s8 data type) and `DynamicQuantizeLinear` operators are only supported on CPU.

安装依赖并运行 `streamlit run web-ui.py` 预览模型效果。由于 ONNXRuntime 算子支持问题，目前仅能够使用 CPU 进行推理。

Codes are released under MIT license.

Model weights are released under the same license as ChatGLM-6b, see [MODEL LICENSE](https://huggingface.co/THUDM/chatglm-6b/blob/main/MODEL_LICENSE).