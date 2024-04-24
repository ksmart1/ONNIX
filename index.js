async function runExample() {
    // Retrieve input values from text boxes
    let x = [];
    x[0] = parseFloat(document.getElementById('box1').value);
    x[1] = parseFloat(document.getElementById('box2').value);
    x[2] = parseFloat(document.getElementById('box3').value);
    x[3] = parseFloat(document.getElementById('box4').value);
    x[4] = parseFloat(document.getElementById('box5').value);
    x[5] = parseFloat(document.getElementById('box6').value);
    x[6] = parseFloat(document.getElementById('box7').value);
    x[7] = parseFloat(document.getElementById('box8').value);
    x[8] = parseFloat(document.getElementById('box9').value);
    x[9] = parseFloat(document.getElementById('box10').value);
    x[10] = parseFloat(document.getElementById('box11').value);
    x[11] = parseFloat(document.getElementById('box12').value);

    // Load scaling parameters
    const response = await fetch('scaling_params.json');
    const scalingParams = await response.json();

    // Scale input features
    for (let i = 0; i < x.length; i++) {
        const feature = `x${i + 1}`;
        console.log("Feature:", feature); // Log the feature name
        console.log("Scaling Params:", scalingParams); // Log the scalingParams object
        const min = scalingParams[feature].min; // This line is causing the error
        const max = scalingParams[feature].max;
        const mean = scalingParams[feature].mean;

    // Perform min-max scaling
    x[i] = (x[i] - mean) / (max - min);
}


    // Create tensor from scaled input
    const tensorX = new onnx.Tensor(x, 'float32', [1, 12]);

    // Load the ONNX model
    const session = new onnx.InferenceSession();
    await session.loadModel("./DLshillOrNw.onnx");

    // Run inference
    const outputMap = await session.run([tensorX]);
    const outputData = outputMap.get('output1');

    // Display inference result
    const predictions = document.getElementById('predictions');
    predictions.innerHTML = `<hr> Classification: ${outputData.data[0].toFixed(2)}`;
}
