const { loadLayersModel } = require("@tensorflow/tfjs");
const tf = require("@tensorflow/tfjs");
async function loadModel() {
  const model = await tf.loadGraphModel("file://saved_model");
  console.log(model);
}
loadModel();
