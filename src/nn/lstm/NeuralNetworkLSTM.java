package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkLSTM {
    private final static int ROWS_COUNT_DEFAULT = 1;
    private final LSTMRow masterRow;
    private final List<LSTMRow> lstmRowList;
    private int rowsCount;

    public NeuralNetworkLSTM(int[] layersCountArray) {
        this.masterRow = new LSTMRow(layersCountArray);
        this.lstmRowList = new ArrayList<LSTMRow>();
        this.lstmRowList.add(masterRow);
        this.rowsCount = ROWS_COUNT_DEFAULT;
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
            this.lstmRowList.get(seriesIndex).setInputToLSTMRow(inputs[seriesIndex]);
        }
    }

    public int getRowsCount() {
        return this.rowsCount;
    }

    public LSTMRow getLastRow() {
        return this.lstmRowList.get(this.rowsCount - 1);
    }

    public List<LSTMRow> getLstmRowList() {
        return this.lstmRowList;
    }

    public void forwardPropogationRow() {
        this.lstmRowList.get(0).forwardPropogationRow();
        for (int rowIndex = 1; rowIndex < this.rowsCount; rowIndex++) {
            for (int cellIndex = 0; cellIndex < this.lstmRowList.get(rowIndex).getLstmCellCount(); cellIndex++) {
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setCellStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getCellState());
                this.lstmRowList.get(rowIndex).getCell(cellIndex).setHiddenStateInput(this.lstmRowList.get(rowIndex - 1).getCell(cellIndex).getHiddenState());
                this.lstmRowList.get(rowIndex).forwardPropogationRow();
            }
        }
    }

}
