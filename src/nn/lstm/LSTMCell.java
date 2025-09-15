package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.Bias;
import nn.common.CommonConstants;
import nn.common.Layer;
import nn.common.Node;

import java.util.List;
import java.util.Random;

public class LSTMCell {
    private List<Node> nodesInput;
    private List<Bias> biases;
    private int inputCount;
    private int nodesCount;
    private int biasesCount;
    private int layerIndex;
    private int gatesNodeCount;
    private Layer forgetGate;
    private Layer inputGate;
    private Layer candidateCellState;
    private Layer outputGate;
    private double[] cellStateInput;
    private double[] cellState;
    private double[] hiddenStateInput;
    private double[] hiddenState;
    private double[] inputVectorX;

    public LSTMCell(int inputCount, int gatesNodeCount, int biasesCount, int layerIndex, String layerName) {
        this.gatesNodeCount = gatesNodeCount;
        this.inputVectorX = new double[inputCount];
        this.cellStateInput = new double[gatesNodeCount];
        this.hiddenStateInput = new double[gatesNodeCount];
        this.forgetGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.inputGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.candidateCellState = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.outputGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
    }

    public int getGatesNodeCount() {
        return this.gatesNodeCount;
    }

    public int getOutputLength() {
        return this.gatesNodeCount;
    }

    public void generatingInitialState(){
        Random randomWeight = new Random();
        for(int i = 0; i < this.hiddenStateInput.length; i++){
            this.hiddenStateInput[i] = randomWeight.nextDouble() * 2 - 1;
            this.cellStateInput[i] = randomWeight.nextDouble() * 2 - 1;
        }
    }

    public void setInitialState(double[] hiddenStateInput, double[] cellStateInput){
        this.hiddenStateInput = hiddenStateInput;
        this.cellStateInput = cellStateInput;
    }

    public void setInputVectorX(double[] inputVectorX){
        if (this.inputVectorX.length != inputVectorX.length){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.inputVectorX = inputVectorX;
    }

    public void setHiddenStateInput(double[] hiddenStateInput){
        if (this.hiddenStateInput.length != hiddenStateInput.length){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.hiddenStateInput = inputVectorX;
    }

    public void setCellStateInput(double[] cellStateInput){
        if (this.cellStateInput.length != cellStateInput.length){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.cellStateInput = cellStateInput;
    }

    public void calculateAllGates(){
        this.forgetGate.calculateLayerSigmaOutputs();
        this.inputGate.calculateLayerSigmaOutputs();
        this.candidateCellState.calculateLayerTanhOutputs();
        this.outputGate.calculateLayerSigmaOutputs();
    }

    public double[] hadamardProduct(double[] a, double[] b){
        if (a.length != b.length){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = a[vectorIndex] * b[vectorIndex];
        }
        return c;
    }

    public double[] pointwiseAddition(double[] a, double[] b){
        if (a.length != b.length){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = a[vectorIndex] + b[vectorIndex];
        }
        return c;
    }

    public double[] tanhFunction(double[] a){
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = (Math.exp(a[vectorIndex]) - Math.exp(-1 * a[vectorIndex])) / (Math.exp(a[vectorIndex]) + Math.exp(-1 * a[vectorIndex]));
        }
        return c;
    }

    public void forwardPropagation(){
        this.calculateAllGates();
        this.cellState = this.pointwiseAddition(this.hadamardProduct(this.forgetGate.getLayerOutputs(), this.cellStateInput), this.hadamardProduct(this.inputGate.getLayerOutputs(), this.candidateCellState.getLayerOutputs()));
        this.hiddenState = this.hadamardProduct(this.outputGate.getLayerOutputs(), this.tanhFunction(this.cellState));
    }

    public double[] getHiddenState(){
        return this.hiddenState;
    }

    public double[] getOutputVector(){
        return this.hiddenState;
    }

    public double[] getCellState(){
        return this.cellState;
    }

    public Layer getForgetGate() {
        return forgetGate;
    }

    public Layer getInputGate() {
        return inputGate;
    }

    public Layer getCandidateCellState() {
        return candidateCellState;
    }

    public Layer getOutputGate() {
        return outputGate;
    }



}
