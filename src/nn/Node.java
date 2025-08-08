package nn;

import java.util.Random;

public class Node {
    private double[] inputs;
    private double[] weights;
    private double sum;
    private double output;
    private int inputCount;
    private double deltaOfNode;
    private double[] deltaOfWeight;

    public Node(int inputCount){
        this.inputCount = inputCount;
        this.inputs = new double[inputCount];
        this.weights = new double[inputCount];
        this.deltaOfWeight = new double[inputCount];
        this.sum = 0.0;
        generateWeights();
    }

    public double[] getInputs() {
        return inputs;
    }

    public double getInput(int inputIndex) {
        return this.inputs[inputIndex];
    }

    public void setInputs(double[] x) {
        this.inputs = x;
    }

    public double getSum() {
        return sum;
    }

    public double getOutput() {
        return output;
    }

    public double[] getWeights() {
        return weights;
    }

    public void generateWeights(){
        Random randomWeight = new Random();
        for(int i = 0; i < this.inputCount; i++){
            this.weights[i] = randomWeight.nextDouble() * 2 - 1;
        }
    }

    public void setCustomWeights(double[] weights){
        this.weights = weights;
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

    public void weightUpdate(){
        for(int weightIndex = 0; weightIndex < this.weights.length; weightIndex++){
            this.weights[weightIndex] = this.weights[weightIndex] - this.deltaOfWeight[weightIndex];
        }
    }

    public void calculateOutput() {
        for(int i = 0; i < this.inputCount; i++){
            this.sum = this.sum + this.inputs[i]*this.weights[i];
        }
        this.output = sigmaActivateFunction(this.sum);
    }

    public double sigmaActivateFunction(double summ){
        return 1.0 / (1.0 + Math.exp(-1 * summ));
    }

    @Override
    public String toString() {
        return String.format("%2.9f", this.output);
    }
}
