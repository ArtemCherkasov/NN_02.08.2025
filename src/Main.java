import nn.helpers.DataHelper;
import nn.helpers.eurusd.MarketPriceEURUSD;
import nn.lstm.NeuralNetworkLSTM;
import nn.simple.NeuralNetworkSimple;

import java.io.IOException;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        String PATH_TO_DATA_DIR = "\\resources\\";
        String FILE_NAME = "\\EURUSD_H1_200906120000_202509251100.csv\\";
        String FILE_WEIGHT_DATA = "\\EURUSD_H1_WEIGHT_35.txt\\";
        int LSTM_CELLS_COUNT_IN_ROW = 35;
        int LSTM_ROW_COUNT = 1;
        int fileDataPointer = 0;
        List<MarketPriceEURUSD> marketPrices;
        double[][] normalMarketPriseSeries;
        NeuralNetworkLSTM neuralNetworkLSTM;

        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
        neuralNetworkLSTM = new NeuralNetworkLSTM(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW)[0].length, LSTM_CELLS_COUNT_IN_ROW, LSTM_ROW_COUNT);
        neuralNetworkLSTM.setNetworkInput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW));
        neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer + 1, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW + 1));
        neuralNetworkLSTM.forwardPropagation();

        filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR).concat(FILE_WEIGHT_DATA);

        String finalFilePath = filePath;
        Runtime.getRuntime().addShutdownHook(new Thread(){
            public void run(){
                System.out.println("Application terminated");
                try {
                    DataHelper.saveLSTMData(finalFilePath, neuralNetworkLSTM);
                    System.out.println("file ".concat(finalFilePath).concat(" has been written"));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        try {
            DataHelper.loadLSTMData(filePath, neuralNetworkLSTM);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        
        for (fileDataPointer = 0; fileDataPointer < 50000; ++fileDataPointer){
            neuralNetworkLSTM.setNetworkInput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW));
            neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer + 1, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW + 1));
            for(int i = 0; i < 1; ++i){
                neuralNetworkLSTM.setDirection();
                neuralNetworkLSTM.learningAction();
                neuralNetworkLSTM.forwardPropagation();
                neuralNetworkLSTM.learningStepValueUpdate();
                System.out.println(i + " " + String.format("%.15f", neuralNetworkLSTM.getMeanSquaredError()));
            }
        }

    }
}