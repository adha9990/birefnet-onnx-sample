const { Tensor } = require("onnxruntime-node");
const sharp = require("sharp");

async function getImageTensorFromPath(imagePath) {
    // Get buffer data from image
    const imageData = await sharp(imagePath)
        .resize(1024, 1024)
        .raw()
        .removeAlpha()
        .toBuffer({ resolveWithObject: true })
        .then(({ data, info }) => data);

    // Create R, G, and B arrays.
    const [redArray, greenArray, blueArray] = new Array(
        new Array(),
        new Array(),
        new Array()
    );

    // Loop through the image buffer and extract the R, G, and B channels
    for (let i = 0; i < imageData.length; i += 3) {
        redArray.push(imageData[i + 0]);
        greenArray.push(imageData[i + 1]);
        blueArray.push(imageData[i + 2]);
    }

    // Concatenate RGB to transpose [width, height, channels] -> [channels, width, height] to a number array
    const transposedData = redArray.concat(greenArray).concat(blueArray);

    // convert and normalize to float32
    const float32Data = new Float32Array(3 * 1024 * 1024);
    for (i = 0; i < transposedData.length; i += 3) {
        float32Data[i + 0] = (transposedData[i + 0] / 255.0 - 0.485) / 0.229;
        float32Data[i + 1] = (transposedData[i + 1] / 255.0 - 0.456) / 0.224;
        float32Data[i + 2] = (transposedData[i + 2] / 255.0 - 0.406) / 0.225;
    }

    // create the tensor object
    const inputTensor = new Tensor("float32", float32Data, [1, 3, 1024, 1024]);

    return inputTensor;
}

module.exports = {
    getImageTensorFromPath,
};
