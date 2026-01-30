package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;
import nn.common.Layer;
import nn.interfaces.LayerInterface;

import java.util.Arrays;
import java.util.Random;

public class LSTMCell implements LayerInterface {
    private final int gatesNodeCount;
    private final Layer forgetGate;
    private final Layer inputGate;
    private final Layer candidateCellState;
    private final Layer outputGate;
    private double[] cellStateInput;
    private double[] cellState;
    private double[] hiddenStateInput;
    private double[] hiddenState;
    private double[] inputVectorX;
    private double[] targetPredictionVector;

    //back propagation fields
    private double[] dEdOutputGate;
    private double[] dEdCellState;
    private double[] dEdHiddenState;
    private double[] dEdInputGate;
    private double[] dEdCandidateCellStateGate;
    private double[] dEdForgetGate;
    private double[] dEdCellStateInput;
    private double[] dEdWInputVectorXOutputGate;
    private double[] dEdWInputVectorXInputGate;
    private double[] dEdWInputVectorXForgetGate;
    private double[] dEdWInputVectorXCandidateCellStateGate;
    private double[] dEdWHiddenStateInputOutputGate;
    private double[] dEdWHiddenStateInputInputGate;
    private double[] dEdWHiddenStateInputForgetGate;
    private double[] dEdWHiddenStateInputCandidateCellStateGate;
    private double[] dEdWBiasOutputGate;
    private double[] dEdWBiasInputGate;
    private double[] dEdWBiasForgetGate;
    private double[] dEdWBiasCandidateCellStateGate;

    public LSTMCell(int inputCount, int gatesNodeCount, int biasesCount, int layerIndex, String layerName) {
        this.gatesNodeCount = gatesNodeCount;
        this.inputVectorX = new double[inputCount];
        this.cellStateInput = new double[gatesNodeCount];
        this.hiddenStateInput = new double[gatesNodeCount];
        this.cellState = new double[gatesNodeCount];
        this.hiddenState = new double[gatesNodeCount];
        this.forgetGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.inputGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.candidateCellState = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
        this.outputGate = new Layer(inputCount + gatesNodeCount, gatesNodeCount, biasesCount, layerIndex);
    }

    public LSTMCell(LSTMCell lstmCell) {
        this.gatesNodeCount = lstmCell.gatesNodeCount;
        this.cellStateInput = new double[lstmCell.cellStateInput.length];
        this.hiddenStateInput = new double[lstmCell.hiddenStateInput.length];
        this.inputVectorX = new double[lstmCell.inputVectorX.length];
        this.cellStateInput = lstmCell.cellStateInput.clone();
        this.hiddenStateInput = lstmCell.hiddenStateInput.clone();
        this.inputVectorX = lstmCell.inputVectorX.clone();
        this.forgetGate = new Layer(lstmCell.forgetGate);
        this.inputGate = new Layer(lstmCell.inputGate);
        this.candidateCellState = new Layer(lstmCell.candidateCellState);
        this.outputGate = new Layer(lstmCell.outputGate);
    }

    public int getGatesNodeCount() {
        return this.gatesNodeCount;
    }

    public int getOutputLength() {
        return this.gatesNodeCount;
    }

    public void generatingInitialState() {
        Random randomWeight = new Random();
        for (int i = 0; i < this.hiddenStateInput.length; i++) {
            this.hiddenStateInput[i] = randomWeight.nextDouble() * 2 - 1;
            this.cellStateInput[i] = randomWeight.nextDouble() * 2 - 1;
        }
    }

    public void setInitialState(double[] hiddenStateInput, double[] cellStateInput) {
        this.hiddenStateInput = new double[hiddenStateInput.length];
        this.cellStateInput = new double[cellStateInput.length];
        this.hiddenStateInput = hiddenStateInput.clone();
        this.cellStateInput = cellStateInput.clone();
    }

    public double[] getCellStateInput() {
        return this.cellStateInput;
    }

    public void setCellStateInput(double[] cellStateInput) {
        if (this.cellStateInput.length != cellStateInput.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }

        this.cellStateInput = cellStateInput.clone();
    }

    public double[] getHiddenStateInput() {
        return this.hiddenStateInput;
    }

    public void setHiddenStateInput(double[] hiddenStateInput) {
        if (this.hiddenStateInput.length != hiddenStateInput.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.hiddenStateInput = hiddenStateInput.clone();
    }

    public double[] getInputVectorX() {
        return this.inputVectorX;
    }

    public void setInputVectorX(double[] inputVectorX) {
        if (this.inputVectorX.length != inputVectorX.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.inputVectorX = inputVectorX.clone();
    }

    public double[] concatenateArrays(double[]... arrays) {
        return Arrays.stream(arrays).flatMapToDouble(val -> Arrays.stream(val)).toArray();
    }

    public void calculateAllGates() {
        this.forgetGate.setInputs(this.concatenateArrays(this.hiddenStateInput, this.inputVectorX));
        this.inputGate.setInputs(this.concatenateArrays(this.hiddenStateInput, this.inputVectorX));
        this.candidateCellState.setInputs(this.concatenateArrays(this.hiddenStateInput, this.inputVectorX));
        this.outputGate.setInputs(this.concatenateArrays(this.hiddenStateInput, this.inputVectorX));
        this.forgetGate.calculateLayerSigmaOutputs();
        this.inputGate.calculateLayerSigmaOutputs();
        this.candidateCellState.calculateLayerTanhOutputs();
        this.outputGate.calculateLayerSigmaOutputs();
    }

    public double[] hadamardProduct(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = a[vectorIndex] * b[vectorIndex];
        }
        return c;
    }

    public double[] pointwiseAddition(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = a[vectorIndex] + b[vectorIndex];
        }
        return c;
    }

    public double[] tanhFunctionVector(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = Math.tanh(a[vectorIndex]);
        }
        return c;
    }

    /**
     * Derivative of the hyperbolic tangent function (1 - tan^2(a))
     *
     * @param a
     * @return double[]
     */
    public double[] derivativeTanhFunctionVector(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = 1.0 - Math.pow(Math.tanh(a[vectorIndex]), 2.0);
        }
        return c;
    }

    /**
     * Derivative of the sigmoid function (a*(1 - a))
     * @param a
     * @return double[]
     */
    public double[] derivativeSigmoidFunctionVector(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = a[vectorIndex] * (1.0 - a[vectorIndex]);
        }
        return c;
    }

    public double[] sigmaFunction(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = 1.0 / (1.0 + Math.exp(-1 * Math.exp(a[vectorIndex])));
        }
        return c;
    }

    public void forwardPropagation() {
        this.calculateAllGates();
        this.cellState = this.hadamardProduct(this.forgetGate.getLayerOutputs(), this.cellStateInput);
        double[] hadamardProductInputGateCandidateGate = this.hadamardProduct(this.inputGate.getLayerOutputs(), this.candidateCellState.getLayerOutputs());
        this.cellState = this.pointwiseAddition(this.cellState, hadamardProductInputGateCandidateGate);
        this.hiddenState = this.hadamardProduct(this.outputGate.getLayerOutputs(), this.tanhFunctionVector(this.cellState));
        this.hiddenState = this.sigmaFunction(this.hiddenState);
    }

    public void cellDerivativesCalculate() {
        this.dEdOutputGate = this.hadamardProduct(this.dEdHiddenState, this.tanhFunctionVector(this.cellState));
        this.dEdCellState = this.hadamardProduct(this.hadamardProduct(this.dEdHiddenState, this.outputGate.getLayerOutputs()), this.derivativeTanhFunctionVector(this.cellState));
        this.dEdInputGate = this.hadamardProduct(this.dEdCellState, this.candidateCellState.getLayerOutputs());
        this.dEdCandidateCellStateGate = this.hadamardProduct(this.dEdCellState, this.inputGate.getLayerOutputs());
        this.dEdForgetGate = this.hadamardProduct(this.dEdCellState, this.cellStateInput);
        this.dEdCellStateInput = this.hadamardProduct(this.dEdCellState, this.forgetGate.getLayerOutputs());
        this.dEdWInputVectorXOutputGate = this.hadamardProduct(this.hadamardProduct(this.dEdOutputGate, this.derivativeSigmoidFunctionVector(this.outputGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputOutputGate = this.hadamardProduct(this.hadamardProduct(this.dEdOutputGate, this.derivativeSigmoidFunctionVector(this.outputGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasOutputGate = this.hadamardProduct(this.dEdOutputGate, this.derivativeSigmoidFunctionVector(this.outputGate.getLayerOutputs()));
        this.dEdWInputVectorXInputGate = this.hadamardProduct(this.hadamardProduct(this.dEdInputGate, this.derivativeSigmoidFunctionVector(this.inputGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputInputGate = this.hadamardProduct(this.hadamardProduct(this.dEdInputGate, this.derivativeSigmoidFunctionVector(this.inputGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasInputGate = this.hadamardProduct(this.dEdInputGate, this.derivativeSigmoidFunctionVector(this.inputGate.getLayerOutputs()));
        this.dEdWInputVectorXForgetGate = this.hadamardProduct(this.hadamardProduct(this.dEdForgetGate, this.derivativeSigmoidFunctionVector(this.forgetGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputForgetGate = this.hadamardProduct(this.hadamardProduct(this.dEdForgetGate, this.derivativeSigmoidFunctionVector(this.forgetGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasForgetGate = this.hadamardProduct(this.dEdForgetGate, this.derivativeSigmoidFunctionVector(this.forgetGate.getLayerOutputs()));
        this.dEdWInputVectorXCandidateCellStateGate = this.hadamardProduct(this.hadamardProduct(this.dEdCandidateCellStateGate, this.derivativeTanhFunctionVector(this.candidateCellState.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputCandidateCellStateGate = this.hadamardProduct(this.hadamardProduct(this.dEdCandidateCellStateGate, this.derivativeTanhFunctionVector(this.candidateCellState.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasCandidateCellStateGate = this.hadamardProduct(this.dEdCandidateCellStateGate, this.derivativeTanhFunctionVector(this.candidateCellState.getLayerOutputs()));
    }

    public double[] getTargetPredictionVector() {
        return this.targetPredictionVector;
    }

    public void setTargetPredictionVector(double[] targetPredictionVector) {
        this.targetPredictionVector = targetPredictionVector;
    }

    public double[] getHiddenState() {
        return this.hiddenState;
    }

    public double[] getOutputVector() {
        return this.hiddenState;
    }

    public double[] getCellState() {
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

    public double[] getdEdOutputGate() {
        return dEdOutputGate;
    }

    public void setdEdOutputGate(double[] dEdOutputGate) {
        this.dEdOutputGate = dEdOutputGate;
    }

    public double[] getdEdCellState() {
        return dEdCellState;
    }

    public void setdEdCellState(double[] dEdCellState) {
        this.dEdCellState = dEdCellState;
    }

    public double[] getdEdHiddenState() {
        return dEdHiddenState;
    }

    public void setdEdHiddenState(double[] dEdHiddenState) {
        this.dEdHiddenState = dEdHiddenState;
    }

    public double[] getdEdInputGate() {
        return dEdInputGate;
    }

    public void setdEdInputGate(double[] dEdInputGate) {
        this.dEdInputGate = dEdInputGate;
    }

    public double[] getdEdCandidateCellStateGate() {
        return dEdCandidateCellStateGate;
    }

    public void setdEdCandidateCellStateGate(double[] dEdCandidateCellStateGate) {
        this.dEdCandidateCellStateGate = dEdCandidateCellStateGate;
    }

    public double[] getdEdForgetGate() {
        return dEdForgetGate;
    }

    public void setdEdForgetGate(double[] dEdForgetGate) {
        this.dEdForgetGate = dEdForgetGate;
    }

    public double[] getdEdCellStateInput() {
        return dEdCellStateInput;
    }

    public void setdEdCellStateInput(double[] dEdCellStateInput) {
        this.dEdCellStateInput = dEdCellStateInput;
    }

    public double[] getdEdWInputVectorXOutputGate() {
        return dEdWInputVectorXOutputGate;
    }

    public void setdEdWInputVectorXOutputGate(double[] dEdWInputVectorXOutputGate) {
        this.dEdWInputVectorXOutputGate = dEdWInputVectorXOutputGate;
    }

    public double[] getdEdWInputVectorXInputGate() {
        return dEdWInputVectorXInputGate;
    }

    public void setdEdWInputVectorXInputGate(double[] dEdWInputVectorXInputGate) {
        this.dEdWInputVectorXInputGate = dEdWInputVectorXInputGate;
    }

    public double[] getdEdWInputVectorXForgetGate() {
        return dEdWInputVectorXForgetGate;
    }

    public void setdEdWInputVectorXForgetGate(double[] dEdWInputVectorXForgetGate) {
        this.dEdWInputVectorXForgetGate = dEdWInputVectorXForgetGate;
    }

    public double[] getdEdWInputVectorXCandidateCellStateGate() {
        return dEdWInputVectorXCandidateCellStateGate;
    }

    public void setdEdWInputVectorXCandidateCellStateGate(double[] dEdWInputVectorXCandidateCellStateGate) {
        this.dEdWInputVectorXCandidateCellStateGate = dEdWInputVectorXCandidateCellStateGate;
    }

    public double[] getdEdWHiddenStateInputOutputGate() {
        return dEdWHiddenStateInputOutputGate;
    }

    public void setdEdWHiddenStateInputOutputGate(double[] dEdWHiddenStateInputOutputGate) {
        this.dEdWHiddenStateInputOutputGate = dEdWHiddenStateInputOutputGate;
    }

    public double[] getdEdWHiddenStateInputInputGate() {
        return dEdWHiddenStateInputInputGate;
    }

    public void setdEdWHiddenStateInputInputGate(double[] dEdWHiddenStateInputInputGate) {
        this.dEdWHiddenStateInputInputGate = dEdWHiddenStateInputInputGate;
    }

    public double[] getdEdWHiddenStateInputForgetGate() {
        return dEdWHiddenStateInputForgetGate;
    }

    public void setdEdWHiddenStateInputForgetGate(double[] dEdWHiddenStateInputForgetGate) {
        this.dEdWHiddenStateInputForgetGate = dEdWHiddenStateInputForgetGate;
    }

    public double[] getdEdWHiddenStateInputCandidateCellStateGate() {
        return dEdWHiddenStateInputCandidateCellStateGate;
    }

    public void setdEdWHiddenStateInputCandidateCellStateGate(double[] dEdWHiddenStateInputCandidateCellStateGate) {
        this.dEdWHiddenStateInputCandidateCellStateGate = dEdWHiddenStateInputCandidateCellStateGate;
    }
}
