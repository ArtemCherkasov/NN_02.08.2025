package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;
import nn.common.Node;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetworkLSTM {
    private final static int ROWS_COUNT_DEFAULT = 1;
    private final LSTMRow masterRow;
    private final List<LSTMRow> lstmRowList;
    private int rowsCount;
    private double currentSquaredError;
    private double learningStepValue = CommonConstants.LEARNING_STEP_DEFAULT_VALUE;

    public NeuralNetworkLSTM(int singleCellInputCount, int cellsCount, int rowsCount) {
        masterRow = null;
        this.rowsCount = rowsCount;
        this.lstmRowList = new ArrayList<LSTMRow>();
        int[] cellsCountArray = new int[cellsCount];
        Arrays.fill(cellsCountArray, singleCellInputCount);
        for (int rowIndex = 0; rowIndex < this.rowsCount; rowIndex++) {
            this.lstmRowList.add(new LSTMRow(cellsCountArray));
        }
    }

    public NeuralNetworkLSTM(LSTMRow lstmRow) {
        this.masterRow = new LSTMRow(lstmRow);
        this.lstmRowList = new ArrayList<LSTMRow>();
        this.lstmRowList.add(masterRow);
        this.rowsCount = this.lstmRowList.size();
    }

    public void addLSTMRowsSeries(int seriesCount) {
        for (int i = 0; i < seriesCount; i++) {
            this.lstmRowList.add(new LSTMRow(this.masterRow));
        }
        this.rowsCount = this.lstmRowList.size();
    }

    public void setInputSeries(double[][] inputs) {
        int seriesCount = inputs.length;
        if (seriesCount != this.lstmRowList.size()) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_SERIES_COUNT, inputs.length, this.lstmRowList.size());
        }

        for (int seriesIndex = 0; seriesIndex < seriesCount; seriesIndex++) {
            this.lstmRowList.get(seriesIndex).setInputToFirsCell(inputs[seriesIndex]);
        }
    }

    public void setNetworkInput(double[][] inputMatrix) {
        for (int cellIndex = 0; cellIndex < this.getLastRow().getLstmCellCount(); ++cellIndex) {
            this.getLastRow().getCellList().get(cellIndex).setInputVectorX(inputMatrix[cellIndex]);
        }
    }

    public double[][] getNetworkOutput() {
        return this.getFirstRow().getRowOutput();
    }

    public int getRowsCount() {
        return this.rowsCount;
    }

    public LSTMRow getLastRow() {
        return this.lstmRowList.get(this.rowsCount - 1);
    }

    public LSTMRow getFirstRow() {
        return this.lstmRowList.get(0);
    }

    public List<LSTMRow> getLstmRowList() {
        return this.lstmRowList;
    }

    /**
    full forward propagation
     */
    public void forwardPropagation() {
        this.lstmRowList.get(0).forwardPropagationRow(0);
        for (int rowIndex = 1; rowIndex < this.rowsCount; rowIndex++) {
            for (int cellIndex = 0; cellIndex < this.lstmRowList.get(rowIndex).getLstmCellCount(); cellIndex++) {
                //TODO
                /*
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setCellStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getCellState());
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setHiddenStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getHiddenState());
                 */
                this.lstmRowList.get(rowIndex).forwardPropagationRow(cellIndex);
            }
        }
    }

    /**
     partial forward propagation for single row
     */
    public void forwardPropagationForSingleRow(int rowIndex, int cellIndex) {
        this.lstmRowList.get(rowIndex).forwardPropagationRow(cellIndex);
    }

    public void setDirection(){
        this.forwardPropagation();
        for (int rowIndex = 0; rowIndex < this.getLstmRowList().size(); rowIndex++) {
            for (int cellIndex = 0; cellIndex < this.getLstmRowList().get(rowIndex).getLstmCellCount(); cellIndex++){
                this.currentSquaredError = this.getMeanSquaredErrorStartFromCellIndex(cellIndex);
                for (int nodeIndex = 0; nodeIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getInputGate().getNodesCount(); ++nodeIndex){
                    for (int weightIndex = 0; weightIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getInputGate().getNode(nodeIndex).getWeights().length; weightIndex++) {
                        this.setNodeDirection(this.getLstmRowList().get(rowIndex).getCell(cellIndex).getInputGate().getNode(nodeIndex), rowIndex, cellIndex, weightIndex);
                    }
                }
                for (int nodeIndex = 0; nodeIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getOutputGate().getNodesCount(); ++nodeIndex){
                    for (int weightIndex = 0; weightIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getOutputGate().getNode(nodeIndex).getWeights().length; weightIndex++) {
                        this.setNodeDirection(this.getLstmRowList().get(rowIndex).getCell(cellIndex).getOutputGate().getNode(nodeIndex), rowIndex, cellIndex, weightIndex);
                    }
                }
                for (int nodeIndex = 0; nodeIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getForgetGate().getNodesCount(); ++nodeIndex){
                    for (int weightIndex = 0; weightIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getForgetGate().getNode(nodeIndex).getWeights().length; weightIndex++) {
                        this.setNodeDirection(this.getLstmRowList().get(rowIndex).getCell(cellIndex).getForgetGate().getNode(nodeIndex), rowIndex, cellIndex, weightIndex);
                    }
                }
                for (int nodeIndex = 0; nodeIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getCandidateCellState().getNodesCount(); ++nodeIndex){
                    for (int weightIndex = 0; weightIndex < this.getLstmRowList().get(rowIndex).getCell(cellIndex).getCandidateCellState().getNode(nodeIndex).getWeights().length; weightIndex++) {
                        this.setNodeDirection(this.getLstmRowList().get(rowIndex).getCell(cellIndex).getCandidateCellState().getNode(nodeIndex), rowIndex, cellIndex, weightIndex);
                    }
                }
            }
        }
    }

    public void learningAction(){
        this.forwardPropagation();
        for (LSTMRow row : this.getLstmRowList()) {
            for (LSTMCell cell : row.getCellList()) {
                for (Node node : cell.getInputGate().getNodes()) {
                    node.learningAction(this.learningStepValue);
                }
                for (Node node : cell.getOutputGate().getNodes()) {
                    node.learningAction(this.learningStepValue);
                }
                for (Node node : cell.getForgetGate().getNodes()) {
                    node.learningAction(this.learningStepValue);
                }
                for (Node node : cell.getCandidateCellState().getNodes()) {
                    node.learningAction(this.learningStepValue);
                }
            }
        }
    }

    public void learningStepValueUpdate(){
        double factor = 1.1;
        while (this.currentSquaredError*factor < 1.0){
            factor = factor*1.1;
        }
        this.learningStepValue = 1.0/factor;
    }

    public void setCurrentSquaredError(double currentSquaredError) {
        this.currentSquaredError = currentSquaredError;
    }

    public double getLearningStepValue() {
        return this.learningStepValue;
    }

    public void setExpectedRowOutput(double[][] expectedRowOutput) {
        this.getFirstRow().setExpectedRowOutput(expectedRowOutput);
    }

    public double[][] getExpectedRowOutput() {
        return this.getFirstRow().getExpectedRowOutput();
    }

    public double getMeanSquaredError() {
        return this.getFirstRow().getMeanSquaredError();
    }

    public double getMeanSquaredErrorStartFromCellIndex(int cellIndex) {
        return this.getFirstRow().getMeanSquaredErrorStartFromCellIndex(cellIndex);
    }

    private void setNodeDirection(Node node, int rowIndex, int cellIndex, int weightIndex){
        node.setNegativeChange(weightIndex);
        this.forwardPropagationForSingleRow(rowIndex, cellIndex);
        double actionSquaredError = this.getMeanSquaredErrorStartFromCellIndex(cellIndex);
        if (actionSquaredError > this.currentSquaredError){
            node.setDirectionPositive(weightIndex);
        } else if (actionSquaredError == this.currentSquaredError){
            node.setDirectionImmutable(weightIndex);
        }
        node.repairWeight(weightIndex);
    }

}
