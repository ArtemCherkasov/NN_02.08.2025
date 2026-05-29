import nn.helpers.DataHelper;
import nn.helpers.eurusd.MarketPriceEURUSD;
import nn.lstm.NeuralNetworkLSTM;
import nn.simple.NeuralNetworkSimple;

import java.io.IOException;
import java.util.Date;
import java.util.List;
import java.util.Random;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        String PATH_TO_DATA_DIR = System.getProperty("file.separator").concat("resources");
        String FILE_NAME = System.getProperty("file.separator").concat("EURUSD_H1_200906120000_202509251100.csv");
        String FILE_WEIGHT_DATA = System.getProperty("file.separator").concat("EURUSD_H1_WEIGHT_35.txt");
        int LSTM_CELLS_COUNT_IN_ROW = 24*10;
        int COUNT_HOURS_OF_FORECASTING = 24*3;
        int LSTM_ROW_COUNT = 1;
        int fileDataPointer = 0;
        List<MarketPriceEURUSD> marketPrices;
        double[][] normalMarketPriseSeries;
        NeuralNetworkLSTM neuralNetworkLSTM;

        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
        neuralNetworkLSTM = new NeuralNetworkLSTM(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW)[0].length, LSTM_CELLS_COUNT_IN_ROW, LSTM_ROW_COUNT);
        neuralNetworkLSTM.setNetworkInput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW));
        neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer + COUNT_HOURS_OF_FORECASTING, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW + COUNT_HOURS_OF_FORECASTING));
        neuralNetworkLSTM.forwardPropagation();

        filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR).concat(FILE_WEIGHT_DATA);

        String finalFilePath = filePath;
        Runtime.getRuntime().addShutdownHook(new Thread(){
            public void run(){
                System.out.println("Application terminated.");
                try {
                    DataHelper.saveLSTMData(finalFilePath, neuralNetworkLSTM);
                    System.out.println("File ".concat(finalFilePath).concat(" has been saved."));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        try {
            DataHelper.loadLSTMData(filePath, neuralNetworkLSTM);
        } catch (IOException e) {
            System.out.println("No such file");
        }
        fileDataPointer = new Random().nextInt(50000);
        while (true){

            neuralNetworkLSTM.setNetworkInput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW));
            neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, fileDataPointer + COUNT_HOURS_OF_FORECASTING, fileDataPointer + LSTM_CELLS_COUNT_IN_ROW + COUNT_HOURS_OF_FORECASTING));
            for(int i = 0; i < 1; ++i){
                neuralNetworkLSTM.setDirection();
                neuralNetworkLSTM.learningAction();
                neuralNetworkLSTM.forwardPropagation();
                neuralNetworkLSTM.learningStepValueUpdate();
                System.out.println(fileDataPointer + " (" + i + ") " + String.format("%.15f", neuralNetworkLSTM.getMeanSquaredError()) + " " + new Date());
            }
            ++fileDataPointer;
        }
    }
}