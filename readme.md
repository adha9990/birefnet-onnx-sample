# Birefnet去背模型-onnx推理範例

從遠端下載 [BiRefNet-DIS-epoch_590.onnx](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx) 模型到本地資料夾。

## 系統要求

- Node.js 版本 ^18.17.0 或 >= 20.3.0

## 安裝依賴

請先執行以下命令安裝所有必要的 npm 依賴：

```bash
npm install
```

## 運行程式

```bash
node main.js
```

## 參考
- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [Classify images in a web application with ONNX Runtime Web
](https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html)
