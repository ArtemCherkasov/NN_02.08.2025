package nn.lstm;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkLSTM {
    private LSTMRow masterRow;

    private List<LSTMRow> lstmRowList;
    private int[] layersCountArray;

    public NeuralNetworkLSTM(int[] layersCountArray) {
        this.layersCountArray = layersCountArray;
        this.masterRow = new LSTMRow(layersCountArray);
        this.lstmRowList = new ArrayList<LSTMRow>();
        this.lstmRowList.add(masterRow);
    }

    public NeuralNetworkLSTM(LSTMRow lstmRow) {
        //TODO copy constructor
    }

    public void addLSTMRowsSeries(int seriesCount){
        for (int i = 0; i < seriesCount; i++) {
            this.lstmRowList.add(this.masterRow);
        }
    }

    public void setInputSeries(double[][] inputs){
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
