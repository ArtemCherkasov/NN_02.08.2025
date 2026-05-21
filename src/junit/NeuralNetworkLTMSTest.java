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

import java.util.List;

class NeuralNetworkLTMSTest {
    private final static double WEIGHT_START_VALUE = 0.005;
    private final static double WEIGHT_STEP_INCREMENT = 0.005;

    private final static String PATH_TO_DATA_DIR = "\\resources\\";
    private final static String FILE_NAME = "\\EURUSD_H1_200906120000_202509251100.csv\\";
    private final static int LSTM_CELLS_COUNT_IN_ROW = 35;
    private final static int LSTM_ROW_COUNT = 1;

    List<MarketPriceEURUSD> marketPrices;
    double[][] normalMarketPriseSeries;
    NeuralNetworkLSTM neuralNetworkLSTM;
    double weight = WEIGHT_START_VALUE;

    @BeforeEach
    public void initNetworkAndLoadData() {
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
        normalMarketPriseSeries = DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, 0, 35);
        neuralNetworkLSTM = new NeuralNetworkLSTM(normalMarketPriseSeries[0].length, LSTM_CELLS_COUNT_IN_ROW, LSTM_ROW_COUNT);
        neuralNetworkLSTM.setNetworkInput(normalMarketPriseSeries);
        neuralNetworkLSTM.setExpectedRowOutput(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, 1, 36));
        for (LSTMRow row : neuralNetworkLSTM.getLstmRowList()) {
            for (LSTMCell cell : row.getCellList()) {
                for (Node node : cell.getInputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, weightGenerate());
                    }
                }
                for (Node node : cell.getOutputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, weightGenerate());
                    }
                }
                for (Node node : cell.getForgetGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, weightGenerate());
                    }
                }
                for (Node node : cell.getCandidateCellState().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, weightGenerate());
                    }
                }
            }
            for (Node node : row.getLastLayer().getNodes()) {
                for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                    node.setCustomWeight(weightIndex, weightGenerate());
                }
            }
        }
        neuralNetworkLSTM.forwardPropagation();
    }

    @Test
    void inputLayersTest() {
        double[] expectedsInput = new double[]{0.12, 0.06, 0.0, 0.05, 0.140954, 0.141129, 0.140929, 0.141069, 0.1002};
        Assertions.assertArrayEquals(expectedsInput, neuralNetworkLSTM.getLastRow().getCell(0).getInputVectorX(), 0.0);
    }

    @Test
    void outputLayersTest() {
        double[][] expectedsInput = new double[][]{{0.5913384287136019, 0.562316925549143, 0.5735756407032029, 0.5846630759029958, 0.5107960908304006, 0.5338735696665052, 0.55777244095302, 0.5800606196362431, 0.5993514341029604}};
        Assertions.assertArrayEquals(expectedsInput[0], neuralNetworkLSTM.getNetworkOutput()[0], 0.0);
    }

    @Test
    void meanSquaredErrorTest(){
        Assertions.assertEquals(0.2730855940757926, neuralNetworkLSTM.getMeanSquaredError());
    }

    private double weightGenerate() {
        this.weight = this.weight + WEIGHT_STEP_INCREMENT;
        if (this.weight > 1.0) {
            this.weight = WEIGHT_START_VALUE;
        }
        return this.weight;
    }

}