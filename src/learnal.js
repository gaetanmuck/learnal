let tanh = (x) => (2 / (1 + Math.exp(-2 * x))) - 1;
let dtanh = (x) => 1 - tanh(x) * tanh(x);
let initW = () => -1 + Math.random() * 2;

class Learnal {

    constructor(inputsNb, neuronNb, layersNb, outputsNb, learningRate) {
        //params
        this.inputsNb = inputsNb;
        this.neuronNb = neuronNb;
        this.layersNb = layersNb - 1; // the inputs layers count as 1
        this.outputsNb = outputsNb;
        this.network = [];
        this.activate = tanh;
        this.learningRate = learningRate;
        //inputs
        let inputsW = new Array(this.neuronNb).fill(new Array(this.inputsNb + 1).fill());
        inputsW = inputsW.map((x) => x.map(initW));
        this.network.push(inputsW);
        //core
        let networkW = new Array(this.layersNb).fill(new Array(this.neuronNb).fill(new Array(this.neuronNb + 1).fill()));
        networkW = networkW.map((x) => x.map((y) => y.map(initW)));
        networkW.map((x) => this.network.push(x));
        //outputs
        let outputsW = new Array(this.outputsNb).fill(new Array(this.neuronNb + 1).fill());
        outputsW = outputsW.map((x) => x.map(initW));
        this.network.push(outputsW);
        //agregations
        this.agregations = new Array(this.network.length + 1).fill(new Array(this.neuronNb).fill());
        this.agregations[0] = new Array(this.inputsNb).fill(0);
        this.agregations[this.agregations.length - 1] = new Array(this.outputsNb).fill(0);
        //activities
        this.activities = new Array(this.network.length + 1).fill(new Array(this.neuronNb).fill());
        this.activities[0] = new Array(this.inputsNb).fill(0);
        this.activities[this.activities.length - 1] = new Array(this.outputsNb).fill(0);
        //errors
        this.errors = new Array(this.network.length + 1).fill(new Array(this.neuronNb).fill(0));
        this.errors[0] = new Array(this.inputsNb).fill(0);
        this.errors[this.errors.length - 1] = new Array(this.outputsNb).fill(0);
    }

    guess(inputs) {
        for (let i = 0; i < inputs.length; i++) {
            this.activities[0][i] = inputs[i];
        }

        for (let i = 1; i < this.agregations.length; i++) {
            for (let j = 0; j < this.agregations[i].length; j++) {
                this.agregations[i][j] = 0;
                for (let k = 0; k < this.activities[i - 1].length; k++) {
                    this.agregations[i][j] += this.network[i - 1][j][k] * this.activities[i - 1][k];
                }
                this.agregations[i][j] += this.network[i - 1][j][this.network[i - 1][j].length - 1]; //biais
                this.activities[i][j] = this.activate(this.agregations[i][j]);
            }
        }

        return this.activities[this.activities.length - 1];
    }


    train(inputs, should) {
        this.guess(inputs);

        //error back propagation
        for (let j = 0; j < should.length; j++) {
            this.errors[this.errors.length - 1][j] = should[j] - this.activities[this.activities.length - 1][j];
        }
        for (let i = this.errors.length - 1 - 1; i >= 1; i--) {
            for (let j = 0; j < this.errors[i].length; j++) {
                this.errors[i][j] = 0;

                let sum = 0;
                for (let k = 0; k < this.errors[i + 1].length; k++) {
                    sum += Math.abs(this.network[i][k][j]);
                }

                for (let k = 0; k < this.errors[i + 1].length; k++) {
                    this.errors[i][j] += this.network[i][k][j] * this.errors[i + 1][k] / sum;
                }
                this.errors[i][j] *= dtanh(this.agregations[i][j]);
            }
        }

        //weights correction
        for (let i = 0; i < this.network.length; i++) {
            for (let j = 0; j < this.network[i].length; j++) {
                for (let k = 0; k < this.network[i][j].length - 1; k++) {
                    this.network[i][j][k] += this.learningRate * this.errors[i + 1][j] * this.activities[i][k];
                }
                this.network[i][j][this.network[i][j].length - 1] += this.learningRate * this.errors[i + 1][j];
            }
        }
    }

}
