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

    public void forwardPropagation() {
        this.lstmRowList.get(0).forwardPropagationRow();
        for (int rowIndex = 1; rowIndex < this.rowsCount; rowIndex++) {
            for (int cellIndex = 0; cellIndex < this.lstmRowList.get(rowIndex).getLstmCellCount(); cellIndex++) {
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setCellStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getCellState());
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setHiddenStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getHiddenState());
                this.lstmRowList.get(rowIndex).forwardPropagationRow();
            }
        }
    }

    public void setDirection(){
        this.forwardPropagation();
        this.currentSquaredError = this.getMeanSquaredError();
        for (LSTMRow row : this.getLstmRowList()) {
            for (int cellIndex = 0; cellIndex < row.getLstmCellCount(); cellIndex++){
                for (Node node : row.getCell(cellIndex).getInputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        this.setNodeDirection(node, cellIndex, weightIndex);
                    }
                }
                for (Node node : row.getCell(cellIndex).getOutputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        this.setNodeDirection(node, cellIndex, weightIndex);
                    }
                }
                for (Node node : row.getCell(cellIndex).getForgetGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        this.setNodeDirection(node, cellIndex, weightIndex);
                    }
                }
                for (Node node : row.getCell(cellIndex).getCandidateCellState().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        this.setNodeDirection(node, cellIndex, weightIndex);
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
        double factor = 1.0;
        while (this.currentSquaredError*factor < 1.0){
            factor = factor*10.0;
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

    private void setNodeDirection(Node node, int cellIndex, int weightIndex){
        node.setNegativeChange(weightIndex);
        this.forwardPropagation();
        double actionSquaredError = this.getMeanSquaredError();
        if (actionSquaredError > this.currentSquaredError){
            node.setDirectionPositive(weightIndex);
        } else if (actionSquaredError == this.currentSquaredError){
            node.setDirectionImmutable(weightIndex);
        }
        node.repairWeight(weightIndex);
    }

}
