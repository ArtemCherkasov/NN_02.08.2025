package junit;

import nn.common.Node;
import nn.helpers.DataHelper;
import nn.helpers.eurusd.MarketPriceEURUSD;
import nn.lstm.LSTMCell;
import nn.lstm.LSTMRow;
import nn.lstm.NeuralNetworkLSTM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;

class NeuralNetworkLTMSTWithRandomWeightsTest {

    private final static String PATH_TO_DATA_DIR = "\\resources\\";
    private final static String FILE_NAME = "\\EURUSD_H1_200906120000_202509251100.csv\\";
    private final static String FILE_WEIGHT_DATA = "\\EURUSD_H1_WEIGHT_35.txt\\";
    private final static int LSTM_CELLS_COUNT_IN_ROW = 35;
    private final static int LSTM_ROW_COUNT = 1;

    List<MarketPriceEURUSD> marketPrices;
    double[][] normalMarketPriseSeries;
    NeuralNetworkLSTM neuralNetworkLSTM;

    @BeforeEach
    public void initNetworkAndLoadData() {
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
        normalMarketPriseSeries = DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, 0, 35);
        neuralNetworkLSTM = new NeuralNetworkLSTM(normalMarketPriseSeries[0].length, LSTM_CELLS_COUNT_IN_ROW, LSTM_ROW_COUNT);
        neuralNetworkLSTM.setNetworkInput(normalMarketPriseSeries);
        neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, 1, 36));
        neuralNetworkLSTM.forwardPropagation();
    }

    @Test
    void meanSquaredErrorTest(){
        //TODO
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR).concat(FILE_WEIGHT_DATA);
        try {
            DataHelper.loadLSTMData(filePath, neuralNetworkLSTM);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        for(int i = 0; i < 60; ++i){
            neuralNetworkLSTM.setDirection();
            neuralNetworkLSTM.learningAction();
            neuralNetworkLSTM.forwardPropagation();
            System.out.println(i + " " + String.format("%.15f", neuralNetworkLSTM.getMeanSquaredError()));
        }
        try {
            DataHelper.saveLSTMData(filePath, neuralNetworkLSTM);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Assertions.assertEquals(0.15275728127875782, neuralNetworkLSTM.getMeanSquaredError());

    }

}