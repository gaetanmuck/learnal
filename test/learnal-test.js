let loaded = false;
let toDisable = document.getElementsByClassName('to-disable');
for (let element of toDisable) {
    element.disabled = true;
}


let fileImages_test, fileLabels_test, fileImages_train, fileLabels_train;
let testImages;
let testImagesLoaded = false;
let trainImages;
let trainImagesLoaded = false;

let canvas = document.getElementById('canvas');
canvas.width = 500;
canvas.height = 500;
let ctx = canvas.getContext('2d');
let imagePicked;

function load() {
    fileImages_test = document.getElementById('images-test').files[0];
    fileLabels_test = document.getElementById('labels-test').files[0];
    fileImages_train = document.getElementById('images-train').files[0];
    fileLabels_train = document.getElementById('labels-train').files[0];

    //load test images
    getMNIST_browser(fileImages_test, fileLabels_test, function(images) {
        testImages = images;
        testImagesLoaded = true;
        if (trainImagesLoaded) {
            console.log('All files loaded');
            for (let element of toDisable) {
                element.disabled = false;
            }
        }
    });

    //load train images
    getMNIST_browser(fileImages_train, fileLabels_train, function(images) {
        trainImages = images;
        trainImagesLoaded = true;
        if (testImagesLoaded) {
            console.log('All files loaded');
            for (let element of toDisable) {
                element.disabled = false;
            }
        }
    });
}


function resetLearnal() {
    console.log("------ RESET LEARNAL ------");
    initialization();
    console.log("==> Done");
}

function testLearnal() {
    console.log("------ TEST LEARNAL ------");
    test();
    console.log("==> Done");
}

function trainLearnal() {
    console.log("------ TRAIN LEARNAL ------");
    train();
    console.log("==> Done");
}


function pickAndDrawImage() {
    imagePicked = testImages[Math.floor(Math.random() * testImages.length)];

    let w = (canvas.width / imagePicked.cols);
    let h = (canvas.height / imagePicked.rows);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < imagePicked.cols; i++) {
        for (let j = 0; j < imagePicked.rows; j++) {
            let color = imagePicked.pixels[j * imagePicked.cols + i];
            ctx.fillStyle = 'rgb(' + color + ',' + color + ',' + color + ')';
            ctx.fillRect(i * w, j * h, w, h);
        }
    }
}

function guessCanvas() {
    let answer = nn.guess(generateInputs(imagePicked));
    let guessing = getAnswer(answer);
    let should = generateShould(imagePicked.label);
    let error = 0;
    for (let i = 0; i < should.length; i++) {
        error += Math.pow(should[i] - answer[i], 2);
    }
    //report
    console.log("---------------");
    console.log(answer);
    console.log(should);
    console.log("Should: " + imagePicked.label);
    console.log("Answer: " + guessing);
    console.log("error : " + error);
}

function trainCanvas() {
    let inputs = generateInputs(imagePicked);
    let should = generateShould(imagePicked.label);
    nn.train(inputs, should);
}


/////////////////////////////////
let nn;

function initialization() {
    nn = new Learnal(28 * 28, 80, 1, 10, 0.2);
}

function test() {
    let trues = 0;
    let falses = 0;

    for (let image of testImages) {
        let inputs = generateInputs(image);
        let guessing = getAnswer(nn.guess(inputs));
        if (guessing == image.label) {
            trues++;
        } else {
            falses++;
        }
    }

    //report
    console.log("trues : " + trues);
    console.log("falses: " + falses);
    console.log("ratio : " + Math.round((trues / (trues + falses)) * 100) + " %");
}

function train() {
    for (let image of trainImages) {
        let inputs = generateInputs(image);
        let should = generateShould(image.label);
        nn.train(inputs, should);
    }
}


function generateInputs(image) {
    let aReturn = new Array(28 * 28).fill(0);
    for (let i = 0; i < image.pixels.length; i++) {
        aReturn[i] = image.pixels[i] / 255;
    }
    return aReturn;
}

function getAnswer(outputs) {
    let sum = outputs.reduce((acc, current) => acc + current);
    let maxIndex = 0;
    let max = outputs[0];
    for (let i = 1; i < outputs.length; i++) {
        if (outputs[i] > max) {
            max = outputs[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

function generateShould(should) {
    let aReturn = new Array(10).fill(-1);
    aReturn[should] = 1;
    return aReturn;
}
