const { InferenceSession } = require("onnxruntime-node");
const { getImageTensorFromPath } = require("./imageHelper");
const sharp = require("sharp");

const modelPath = "./BiRefNet-DIS-epoch_590.onnx";
const imagePath = "./input.png";

(async () => {
    const { data, info } = await sharp(imagePath)
        .raw()
        .toBuffer({ resolveWithObject: true });

    console.time("Image Preprocessing Time");
    const imageTensor = await getImageTensorFromPath(imagePath);
    console.timeEnd("Image Preprocessing Time");

    console.time("Model Load Time");
    const session = await InferenceSession.create(modelPath);
    console.timeEnd("Model Load Time");

    console.time("Inference Time");
    const outputData = await session.run({ input_image: imageTensor });
    console.timeEnd("Inference Time");

    console.time("Post Processing Time");
    const outputImage = outputData[session.outputNames[0]].data;

    // Binarization
    const normOutputImage = outputImage.map((value) => (value <= 0 ? 0 : 255));

    // Resize mask image to original size
    const maskImageData = await sharp(Buffer.from(normOutputImage), {
        raw: {
            width: 1024,
            height: 1024,
            channels: 1,
        },
    })
        .resize({
            width: info.width,
            height: info.height,
            fit: sharp.fit.contain,
        })
        .ensureAlpha()
        .raw()
        .toBuffer();

    // Combine original image with mask image
    const newImageBuffer = new Float32Array(4 * info.width * info.height);
    for (let i = 0; i < newImageBuffer.length; i += 4) {
        newImageBuffer[i + 0] = data[i + 0];
        newImageBuffer[i + 1] = data[i + 1];
        newImageBuffer[i + 2] = data[i + 2];
        newImageBuffer[i + 3] = maskImageData[i];
    }
    console.timeEnd("Post Processing Time");

    // Save the new image
    await sharp(Buffer.from(newImageBuffer), {
        raw: {
            width: info.width,
            height: info.height,
            channels: 4,
        },
    })
        .ensureAlpha()
        .toFile("./output.png");
})();
