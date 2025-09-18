package nn.common;

import exceptions.NNInputExceptions;

import java.util.Random;

public class Node {
    private final int inputCount;
    private double[] inputs;
    private double[] weights;
    private double sum;
    private double nodeValue;
    private double deltaOfNode; // dE_dOut*dOut_dNet
    private double[] deltaOfWeight;

    public Node(int inputCount) {
        this.inputCount = inputCount;
        this.inputs = new double[inputCount];
        this.weights = new double[inputCount];
        this.deltaOfWeight = new double[inputCount];
        this.sum = 0.0;
        generateWeights();
    }

    public Node(Node node) {
        this.inputs = node.inputs;
        this.weights = node.weights;
        this.sum = node.sum;
        this.nodeValue = node.nodeValue;
        this.inputCount = node.inputCount;
        this.deltaOfNode = node.deltaOfNode;
        this.deltaOfWeight = node.deltaOfWeight;
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] x) {
        this.inputs = x;
    }

    public double getInput(int inputIndex) {
        return this.inputs[inputIndex];
    }

    public double getSum() {
        return sum;
    }

    public double getNodeValue() {
        return nodeValue;
    }

    public void setNodeValue(double nodeValue) {
        this.nodeValue = nodeValue;
    }

    public double[] getWeights() {
        return weights;
    }

    public double getWeight(int weightIndex) {
        return this.weights[weightIndex];
    }

    public void generateWeights() {
        Random randomWeight = new Random();
        for (int i = 0; i < this.inputCount; i++) {
            this.weights[i] = randomWeight.nextDouble() * 2 - 1;
        }
    }

    public void setCustomWeights(double[] weights) {
        if (this.weights.length != weights.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT, weights.length, this.weights.length);
        }
        this.weights = weights;
    }

    public void setCustomWeight(int weigthIndex, double weight) {
        this.weights[weigthIndex] = weight;
    }

    public double getDeltaOfNode() {
        return deltaOfNode;
    }

    public void setDeltaOfNode(double deltaOfNode) {
        this.deltaOfNode = deltaOfNode;
    }

    public double[] getDeltaOfWeight() {
        return deltaOfWeight;
    }

    public void setDeltaOfWeight(double[] deltaOfWeight) {
        this.deltaOfWeight = deltaOfWeight;
    }

    public void weightUpdate() {
        for (int weightIndex = 0; weightIndex < this.weights.length; weightIndex++) {
            this.weights[weightIndex] = this.weights[weightIndex] - this.deltaOfWeight[weightIndex];
        }
    }

    public void calculateSigmaOutput() {
        this.sum = 0.0;
        for (int i = 0; i < this.inputCount; i++) {
            this.sum = this.sum + this.inputs[i] * this.weights[i];
        }
        this.nodeValue = sigmaActivateFunction(this.sum);
    }

    public void calculateHiberbolicTangentOutput() {
        this.sum = 0.0;
        for (int i = 0; i < this.inputCount; i++) {
            this.sum = this.sum + this.inputs[i] * this.weights[i];
        }
        this.nodeValue = tanhActivateFunction(this.sum);
    }

    public double sigmaActivateFunction(double summ) {
        return 1.0 / (1.0 + Math.exp(-1 * summ));
    }

    public double tanhActivateFunction(double summ) {
        return (Math.exp(summ) - Math.exp(-1 * summ)) / (Math.exp(summ) + Math.exp(-1 * summ));
    }

    @Override
    public String toString() {
        return String.format("%2.9f", this.nodeValue);
    }
}
