package nn.lstm;

import nn.common.CommonConstants;
import nn.common.Layer;

import java.util.ArrayList;
import java.util.List;

public class LSTMRow {
    private final int lstmCellCount;
    List<LSTMCell> cellList;
    double[][] rowOutput;
    double[][] expectedRowOutput;
    @Deprecated
    Layer lastLayer;

    public LSTMRow(int[] inputsCountArray) {
        this.lstmCellCount = inputsCountArray.length;
        this.cellList = new ArrayList<LSTMCell>();
        this.cellList.add(new LSTMCell(inputsCountArray[0], inputsCountArray[0], 1, 0, CommonConstants.LSTM_CELL_NAME));
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; cellIndex++) {
            this.cellList.add(new LSTMCell(this.cellList.get(cellIndex - 1).getOutputLength(), inputsCountArray[cellIndex], 1, cellIndex, CommonConstants.LSTM_CELL_NAME));
        }
        this.lastLayer = new Layer(this.getLastLSTMCellOutput().length, this.getLastLSTMCellOutput().length, 1, this.getLstmCellCount());
        this.rowOutput = new double[lstmCellCount][this.getLastLSTMCellOutput().length];
        this.expectedRowOutput = new double[lstmCellCount][this.getLastLSTMCellOutput().length];
    }

    public LSTMRow(LSTMRow lstmRow) {
        this.lstmCellCount = lstmRow.lstmCellCount;
        this.cellList = new ArrayList<LSTMCell>();
        for (LSTMCell lstmCell : lstmRow.cellList) {
            this.cellList.add(new LSTMCell(lstmCell));
        }
        this.lastLayer = new Layer(lstmRow.getLastLayer());
        this.rowOutput = new double[lstmCellCount][lstmRow.getCell(0).getOutputLength()];
        this.expectedRowOutput = new double[lstmCellCount][lstmRow.getCell(0).getOutputLength()];
    }

    public List<LSTMCell> getCellList() {
        return this.cellList;
    }

    public LSTMCell getCell(int cellIndex) {
        return this.cellList.get(cellIndex);
    }

    public void setInputToFirsCell(double[] inputVector) {
        this.getCell(CommonConstants.FIRST_CELL).setInputVectorX(inputVector);
    }

    public void forwardPropagationRow() {
        this.getCell(CommonConstants.FIRST_CELL).forwardPropagation();
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; ++cellIndex) {
            double[] prevHiddenState = this.getCell(cellIndex - 1).getHiddenState();
            double[] prevCellSate = this.getCell(cellIndex - 1).getCellState();
            this.getCell(cellIndex).setHiddenStateInput(prevHiddenState);
            this.getCell(cellIndex).setCellStateInput(prevCellSate);
            this.getCell(cellIndex).forwardPropagation();
        }
        double[] outputLastVector = this.getCell(this.getLstmCellCount() - 1).getOutputVector();
        this.lastLayer.setInputs(outputLastVector);
        this.lastLayer.calculateLayerSigmaOutputs();
    }

    public double[] getLastLSTMCellOutput() {
        return this.getCell(this.lstmCellCount - 1).getOutputVector();
    }

    public Layer getLastLayer() {
        return this.lastLayer;
    }

    public int getLstmCellCount() {
        return this.lstmCellCount;
    }

    public double[][] getRowOutput() {
        for (int cellIndex = 0; cellIndex < lstmCellCount; ++cellIndex) {
            this.rowOutput[cellIndex] = this.cellList.get(cellIndex).getOutputVector();
        }
        return this.rowOutput;
    }

    public void setExpectedRowOutput(double[][] expectedRowOutput) {
        for (int cellIndex = 0; cellIndex < lstmCellCount; ++cellIndex) {
            this.cellList.get(cellIndex).setExpectedVector(expectedRowOutput[cellIndex]);
        }
    }

    public double[][] getExpectedRowOutput() {
        for (int cellIndex = 0; cellIndex < lstmCellCount; ++cellIndex) {
            this.expectedRowOutput[cellIndex] = this.cellList.get(cellIndex).getExpectedVector();
        }
        return this.expectedRowOutput;
    }

    public double getMeanSquaredError() {
        double[][] tagetMatrix = this.getExpectedRowOutput();
        double[][] predictedMatrix = this.getRowOutput();
        int cellsCount = this.lstmCellCount;
        int outputCountPerCell = this.getCell(0).getOutputLength();
        double mse = 0.0;
        int totalElementCount = cellsCount * outputCountPerCell;
        for(int cellIndex = 0; cellIndex < cellsCount; ++cellIndex){
            for (int outputCount = 0; outputCount < outputCountPerCell; ++outputCount){
                mse = mse + Math.pow(tagetMatrix[cellIndex][outputCount] - predictedMatrix[cellIndex][outputCount], 2);
            }
        }
        mse = mse / totalElementCount;
        return mse;
    }

    public double getMeanSquaredErrorStartFromCellIndex(int index) {
        double[][] tagetMatrix = this.getExpectedRowOutput();
        double[][] predictedMatrix = this.getRowOutput();
        int cellsCount = this.lstmCellCount;
        int outputCountPerCell = this.getCell(0).getOutputLength();
        double mse = 0.0;
        int totalElementCount = cellsCount * outputCountPerCell;
        for(int cellIndex = index; cellIndex < cellsCount; ++cellIndex){
            for (int outputCount = 0; outputCount < outputCountPerCell; ++outputCount){
                mse = mse + Math.pow(tagetMatrix[cellIndex][outputCount] - predictedMatrix[cellIndex][outputCount], 2);
            }
        }
        mse = mse / totalElementCount;
        return mse;
    }
}
