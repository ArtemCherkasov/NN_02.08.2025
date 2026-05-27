package nn.common;

import enums.Direction;
import exceptions.NNInputExceptions;

import java.util.Random;

public class Node {
    private final static int SINGLE_INPUT_COUNT = 1;
    private final static double SIMPLE_UNIT_WEIGHTS = 1.0;
    private final int inputCount;
    private double[] inputs;
    private double[] weights;
    private double tempWeigth;
    private double sum;
    private double nodeValue;
    private double deltaOfNode; // dE_dOut*dOut_dNet
    private Direction[] directionOfChange;
    private double[] deltaOfWeight;

    public Node() {
        this.inputCount = SINGLE_INPUT_COUNT;
        this.inputs = new double[SINGLE_INPUT_COUNT];
        this.weights = new double[SINGLE_INPUT_COUNT];
        this.directionOfChange = new Direction[SINGLE_INPUT_COUNT];
        this.deltaOfWeight = new double[SINGLE_INPUT_COUNT];
        this.sum = 0.0;
        generateSimpleUnitWeights();
    }

    public Node(int inputCount) {
        this.inputCount = inputCount;
        this.inputs = new double[inputCount];
        this.weights = new double[inputCount];
        this.directionOfChange = new Direction[inputCount];
        this.deltaOfWeight = new double[inputCount];
        this.sum = 0.0;
        generateWeights();
        for(int inputIndex = 0; inputIndex < inputCount; ++inputIndex){
            this.directionOfChange[inputIndex] = Direction.IMMUTABLE;
        }
    }

    public Node(Node node) {
        this.inputs = new double[node.inputs.length];
        this.inputs = node.inputs.clone();
        this.weights = new double[node.weights.length];
        this.weights = node.weights.clone();
        this.sum = node.sum;
        this.nodeValue = node.nodeValue;
        this.inputCount = node.inputCount;
        this.deltaOfNode = node.deltaOfNode;
        this.deltaOfWeight = new double[node.deltaOfWeight.length];
        this.deltaOfWeight = node.deltaOfWeight.clone();
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] x) {
        this.inputs = x.clone();
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
            this.weights[i] = randomWeight.nextDouble();
        }
    }

    public void generateSimpleUnitWeights() {
        for (int i = 0; i < this.inputCount; i++) {
            this.weights[i] = SIMPLE_UNIT_WEIGHTS;
        }
    }

    public void setCustomWeights(double[] weights) {
        if (this.weights.length != weights.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT, weights.length, this.weights.length);
        }
        this.weights = weights.clone();
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

    public void learningAction(double learningStep){
        for(int directionOfChangeIndex = 0; directionOfChangeIndex < this.inputCount; ++directionOfChangeIndex){
            switch (this.directionOfChange[directionOfChangeIndex]){
                case POSITIVE :
                    this.weights[directionOfChangeIndex] = this.weights[directionOfChangeIndex] + learningStep;
                    break;
                case NEGATIVE:
                    this.weights[directionOfChangeIndex] = this.weights[directionOfChangeIndex] - learningStep;
                    break;
            }
        }
    }

    public void setPositiveChange(int directionOfChangeIndex){
        this.tempWeigth = this.weights[directionOfChangeIndex];
        this.directionOfChange[directionOfChangeIndex] = Direction.POSITIVE;
        this.weights[directionOfChangeIndex] = this.weights[directionOfChangeIndex] + CommonConstants.LEARNING_STEP_DEFAULT_VALUE;
    }

    public void setNegativeChange(int directionOfChangeIndex){
        this.tempWeigth = this.weights[directionOfChangeIndex];
        this.directionOfChange[directionOfChangeIndex] = Direction.NEGATIVE;
        this.weights[directionOfChangeIndex] = this.weights[directionOfChangeIndex] - CommonConstants.LEARNING_STEP_DEFAULT_VALUE;
    }

    public void setDirectionImmutable(int directionOfChangeIndex){
        this.directionOfChange[directionOfChangeIndex] = Direction.IMMUTABLE;
    }

    public void setDirectionPositive(int directionOfChangeIndex){
        this.directionOfChange[directionOfChangeIndex] = Direction.POSITIVE;
    }

    public void setDirectionNegative(int directionOfChangeIndex){
        this.directionOfChange[directionOfChangeIndex] = Direction.NEGATIVE;
    }

    public void repairConditionWithDirection(int directionOfChangeIndex){
        this.weights[directionOfChangeIndex] = this.tempWeigth;
        this.directionOfChange[directionOfChangeIndex] = Direction.IMMUTABLE;
    }

    public void repairWeight(int directionOfChangeIndex){
        this.weights[directionOfChangeIndex] = this.tempWeigth;
    }

    @Override
    public String toString() {
        return String.format("%2.9f", this.nodeValue);
    }
}
