package nn.lstm;

import nn.common.CommonConstants;
import nn.common.Layer;

import java.util.ArrayList;
import java.util.List;

public class LSTMRow {
    private final int lstmCellCount;
    List<LSTMCell> cellList;
    Layer lastLayer;

    public LSTMRow(int[] layersCountArray) {
        this.lstmCellCount = layersCountArray.length;
        this.cellList = new ArrayList<LSTMCell>();
        this.cellList.add(new LSTMCell(layersCountArray[0], layersCountArray[0], 1, 0, CommonConstants.LSTM_CELL_NAME));
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; cellIndex++) {
            this.cellList.add(new LSTMCell(this.cellList.get(cellIndex - 1).getOutputLength(), layersCountArray[cellIndex], 1, cellIndex, CommonConstants.LSTM_CELL_NAME));
        }
        this.lastLayer = new Layer(this.getLastLSTMCellOutput().length, this.getLastLSTMCellOutput().length, 1, this.getLstmCellCount());
    }

    public LSTMRow(LSTMRow lstmRow) {
        this.lstmCellCount = lstmRow.lstmCellCount;
        this.cellList = new ArrayList<LSTMCell>();
        for (LSTMCell lstmCell : lstmRow.cellList) {
            this.cellList.add(new LSTMCell(lstmCell));
        }
        this.lastLayer = new Layer(lstmRow.getLastLayer());
    }

    public List<LSTMCell> getCellList() {
        return this.cellList;
    }

    public LSTMCell getCell(int cellIndex) {
        return this.cellList.get(cellIndex);
    }

    public void setInputToLSTMRow(double[] inputVector) {
        this.getCell(CommonConstants.FIRST_CELL).setInputVectorX(inputVector);
    }

    public void forwardPropagationRow() {
        this.getCell(CommonConstants.FIRST_CELL).forwardPropagation();
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; cellIndex++) {
            double[] outputVector = this.getCell(cellIndex - 1).getOutputVector();
            this.getCell(cellIndex).setInputVectorX(outputVector);
            this.getCell(cellIndex).forwardPropagation();
        }
        double[] outputLastVector = this.getCell(this.getLstmCellCount() - 1).getOutputVector();
        this.lastLayer.setInputs(outputLastVector);
        this.lastLayer.calculateLayerSigmaOutputs();
    }

    public double[] getLastLSTMCellOutput(){
        return this.getCell(this.lstmCellCount - 1).getOutputVector();
    }

    public double[] getLastLayerOutput(){
        return this.lastLayer.getLayerOutputs();
    }

    public Layer getLastLayer() {
        return this.lastLayer;
    }

    public int getLstmCellCount() {
        return this.lstmCellCount;
    }
}
