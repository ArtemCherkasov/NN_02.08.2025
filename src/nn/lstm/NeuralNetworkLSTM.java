package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkLSTM {
    private final LSTMRow masterRow;
    private final List<LSTMRow> lstmRowList;

    public NeuralNetworkLSTM(int[] layersCountArray) {
        this.masterRow = new LSTMRow(layersCountArray);
        this.lstmRowList = new ArrayList<LSTMRow>();
        this.lstmRowList.add(masterRow);
    }

    public NeuralNetworkLSTM(LSTMRow lstmRow) {
        this.masterRow = new LSTMRow(lstmRow);
        this.lstmRowList = new ArrayList<LSTMRow>();
        this.lstmRowList.add(masterRow);
    }

    public void addLSTMRowsSeries(int seriesCount) {
        for (int i = 0; i < seriesCount; i++) {
            this.lstmRowList.add(new LSTMRow(this.masterRow));
        }
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

    public List<LSTMRow> getLstmRowList() {
        return lstmRowList;
    }

}
