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

    //back propagation fields
    private double[] dEdOutput;
    private double[] dEdCellState;
    private double[] dEdHiddenState;
    private double[] dEdInput;
    private double[] dEdCandidateCellState;
    private double[] dEdForget;
    private double[] dEdCellStateInput;
    private double[] dEdWInputVectorXOutput;
    private double[] dEdWInputVectorXInput;
    private double[] dEdWInputVectorXForget;
    private double[] dEdWInputVectorXCandidate;
    private double[] dEdWHiddenStateInputOutput;
    private double[] dEdWHiddenStateInputInput;
    private double[] dEdWHiddenStateInputForget;
    private double[] dEdWHiddenStateInputCandidate;
    private double[] dEdWBiasOutput;
    private double[] dEdWBiasInput;
    private double[] dEdWBiasForget;
    private double[] dEdWBiasCandidate;

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

    public double[] tanhFunction(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = (Math.exp(a[vectorIndex]) - Math.exp(-1 * a[vectorIndex])) / (Math.exp(a[vectorIndex]) + Math.exp(-1 * a[vectorIndex]));
        }
        return c;
    }

    /**
     * function (1 - tan^2(a))
     *
     * @param a
     * @return double[]
     */
    public double[] oneSubtractTanhFunctionSqr(double[] a) {
        int vectorLength = a.length;
        double[] c = new double[vectorLength];
        for (int vectorIndex = 0; vectorIndex < vectorLength; vectorIndex++) {
            c[vectorIndex] = 1.0 - Math.pow((Math.exp(a[vectorIndex]) - Math.exp(-1 * a[vectorIndex])) / (Math.exp(a[vectorIndex]) + Math.exp(-1 * a[vectorIndex])), 2.0);
        }
        return c;
    }

    public double[] subtractSquareOfXfromX(double[] a) {
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
        this.hiddenState = this.hadamardProduct(this.outputGate.getLayerOutputs(), this.tanhFunction(this.cellState));
        this.hiddenState = this.sigmaFunction(this.hiddenState);
    }

    public void cellDerivativesCalculate() {
        this.dEdOutput = this.hadamardProduct(this.dEdHiddenState, this.tanhFunction(this.cellState));
        this.dEdCellState = this.hadamardProduct(this.hadamardProduct(this.dEdHiddenState, this.outputGate.getLayerOutputs()), this.oneSubtractTanhFunctionSqr(this.cellState));
        this.dEdInput = this.hadamardProduct(this.dEdCellState, this.candidateCellState.getLayerOutputs());
        this.dEdCandidateCellState = this.hadamardProduct(this.dEdCellState, this.inputGate.getLayerOutputs());
        this.dEdForget = this.hadamardProduct(this.dEdCellState, this.cellStateInput);
        this.dEdCellStateInput = this.hadamardProduct(this.dEdCellState, this.forgetGate.getLayerOutputs());
        this.dEdWInputVectorXOutput = this.hadamardProduct(this.hadamardProduct(this.dEdOutput, this.subtractSquareOfXfromX(this.outputGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputOutput = this.hadamardProduct(this.hadamardProduct(this.dEdOutput, this.subtractSquareOfXfromX(this.outputGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasOutput = this.hadamardProduct(this.dEdOutput, this.subtractSquareOfXfromX(this.outputGate.getLayerOutputs()));
        this.dEdWInputVectorXInput = this.hadamardProduct(this.hadamardProduct(this.dEdInput, this.subtractSquareOfXfromX(this.inputGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputInput = this.hadamardProduct(this.hadamardProduct(this.dEdInput, this.subtractSquareOfXfromX(this.inputGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasInput = this.hadamardProduct(this.dEdInput, this.subtractSquareOfXfromX(this.inputGate.getLayerOutputs()));
        this.dEdWInputVectorXForget = this.hadamardProduct(this.hadamardProduct(this.dEdForget, this.subtractSquareOfXfromX(this.forgetGate.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputForget = this.hadamardProduct(this.hadamardProduct(this.dEdForget, this.subtractSquareOfXfromX(this.forgetGate.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasForget = this.hadamardProduct(this.dEdForget, this.subtractSquareOfXfromX(this.forgetGate.getLayerOutputs()));
        this.dEdWInputVectorXCandidate = this.hadamardProduct(this.hadamardProduct(this.dEdCandidateCellState, this.subtractSquareOfXfromX(this.candidateCellState.getLayerOutputs())), this.inputVectorX);
        this.dEdWHiddenStateInputCandidate = this.hadamardProduct(this.hadamardProduct(this.dEdCandidateCellState, this.subtractSquareOfXfromX(this.candidateCellState.getLayerOutputs())), this.hiddenStateInput);
        this.dEdWBiasCandidate = this.hadamardProduct(this.dEdCandidateCellState, this.subtractSquareOfXfromX(this.candidateCellState.getLayerOutputs()));
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

    public double[] getdEdOutput() {
        return dEdOutput;
    }

    public void setdEdOutput(double[] dEdOutput) {
        this.dEdOutput = dEdOutput;
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

    public double[] getdEdInput() {
        return dEdInput;
    }

    public void setdEdInput(double[] dEdInput) {
        this.dEdInput = dEdInput;
    }

    public double[] getdEdCandidateCellState() {
        return dEdCandidateCellState;
    }

    public void setdEdCandidateCellState(double[] dEdCandidateCellState) {
        this.dEdCandidateCellState = dEdCandidateCellState;
    }

    public double[] getdEdForget() {
        return dEdForget;
    }

    public void setdEdForget(double[] dEdForget) {
        this.dEdForget = dEdForget;
    }

    public double[] getdEdCellStateInput() {
        return dEdCellStateInput;
    }

    public void setdEdCellStateInput(double[] dEdCellStateInput) {
        this.dEdCellStateInput = dEdCellStateInput;
    }

    public double[] getdEdWInputVectorXOutput() {
        return dEdWInputVectorXOutput;
    }

    public void setdEdWInputVectorXOutput(double[] dEdWInputVectorXOutput) {
        this.dEdWInputVectorXOutput = dEdWInputVectorXOutput;
    }

    public double[] getdEdWInputVectorXInput() {
        return dEdWInputVectorXInput;
    }

    public void setdEdWInputVectorXInput(double[] dEdWInputVectorXInput) {
        this.dEdWInputVectorXInput = dEdWInputVectorXInput;
    }

    public double[] getdEdWInputVectorXForget() {
        return dEdWInputVectorXForget;
    }

    public void setdEdWInputVectorXForget(double[] dEdWInputVectorXForget) {
        this.dEdWInputVectorXForget = dEdWInputVectorXForget;
    }

    public double[] getdEdWInputVectorXCandidate() {
        return dEdWInputVectorXCandidate;
    }

    public void setdEdWInputVectorXCandidate(double[] dEdWInputVectorXCandidate) {
        this.dEdWInputVectorXCandidate = dEdWInputVectorXCandidate;
    }

    public double[] getdEdWHiddenStateInputOutput() {
        return dEdWHiddenStateInputOutput;
    }

    public void setdEdWHiddenStateInputOutput(double[] dEdWHiddenStateInputOutput) {
        this.dEdWHiddenStateInputOutput = dEdWHiddenStateInputOutput;
    }

    public double[] getdEdWHiddenStateInputInput() {
        return dEdWHiddenStateInputInput;
    }

    public void setdEdWHiddenStateInputInput(double[] dEdWHiddenStateInputInput) {
        this.dEdWHiddenStateInputInput = dEdWHiddenStateInputInput;
    }

    public double[] getdEdWHiddenStateInputForget() {
        return dEdWHiddenStateInputForget;
    }

    public void setdEdWHiddenStateInputForget(double[] dEdWHiddenStateInputForget) {
        this.dEdWHiddenStateInputForget = dEdWHiddenStateInputForget;
    }

    public double[] getdEdWHiddenStateInputCandidate() {
        return dEdWHiddenStateInputCandidate;
    }

    public void setdEdWHiddenStateInputCandidate(double[] dEdWHiddenStateInputCandidate) {
        this.dEdWHiddenStateInputCandidate = dEdWHiddenStateInputCandidate;
    }
}
