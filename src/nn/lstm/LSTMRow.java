package nn.lstm;

import nn.common.CommonConstants;

import java.util.ArrayList;
import java.util.List;

public class LSTMRow {
    List<LSTMCell> cellList;
    private int lstmCellCount;

    public LSTMRow(int[] layersCountArray) {
        this.lstmCellCount = layersCountArray.length;
        this.cellList = new ArrayList<LSTMCell>();
        this.cellList.add(new LSTMCell(layersCountArray[0], layersCountArray[0], 1, 0, CommonConstants.LSTM_CELL_NAME));
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; cellIndex++) {
            this.cellList.add(new LSTMCell(this.cellList.get(cellIndex - 1).getOutputLength(), layersCountArray[cellIndex], 1, 0, CommonConstants.LSTM_CELL_NAME));
        }
    }

    public LSTMRow(LSTMRow lstmRow) {
        //TODO copy constructor
    }

    public List<LSTMCell> getCellList() {
        return this.cellList;
    }

    public LSTMCell getCell(int cellIndex) {
        return this.cellList.get(cellIndex);
    }

    public void setInputToLSTMRow(double[] inputVector){
        this.getCell(CommonConstants.FIRST_CELL).setInputVectorX(inputVector);
    }

    public void forwardPropogationRow(){
        this.getCell(CommonConstants.FIRST_CELL).forwardPropagation();
        for (int cellIndex = 1; cellIndex < this.lstmCellCount; cellIndex++) {
            this.getCell(cellIndex).setInputVectorX(this.getCell(cellIndex - 1).getHiddenState());
            this.getCell(cellIndex).forwardPropagation();
        }
    }

}
