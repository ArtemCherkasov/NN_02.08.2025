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

class NeuralNetworkLTMSTWithCustomWeightsTest {
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
        double[] expectedInput = new double[]{0.12, 0.06, 0.0, 0.05, 0.140954, 0.141129, 0.140929, 0.141069, 0.1002};
        Assertions.assertArrayEquals(expectedInput, neuralNetworkLSTM.getLastRow().getCell(0).getInputVectorX(), 0.0);
    }

    @Test
    void outputLayersTest() {
        double[][] expectedOutput = new double[][]{{0.36950113148288527, 0.25057054125561595, 0.2964548219636936, 0.3419456865478178, 0.04319107637075797, 0.13570214294740843, 0.23212647822513555, 0.32302224629996096, 0.40276347863714385}};
        Assertions.assertArrayEquals(expectedOutput[0], neuralNetworkLSTM.getNetworkOutput()[0], 0.0);
    }

    @Test
    void meanSquaredErrorTest(){
        for(int i = 0; i < 2; ++i){
            neuralNetworkLSTM.setDirection();
            for (int j = 0; j < 2; ++j){
                neuralNetworkLSTM.learningAction();
                neuralNetworkLSTM.forwardPropagation();
                System.out.println(neuralNetworkLSTM.getMeanSquaredError());
            }
        }
        Assertions.assertEquals(0.6256966863807255, neuralNetworkLSTM.getMeanSquaredError());
    }

    @Test
    void learningRateUpdateTest(){
        neuralNetworkLSTM.setCurrentSquaredError(0.12);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.1, neuralNetworkLSTM.getLearningStepValue());
        neuralNetworkLSTM.setCurrentSquaredError(0.18);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.1, neuralNetworkLSTM.getLearningStepValue());
        neuralNetworkLSTM.setCurrentSquaredError(0.67123);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.1, neuralNetworkLSTM.getLearningStepValue());
        neuralNetworkLSTM.setCurrentSquaredError(0.067123);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.01, neuralNetworkLSTM.getLearningStepValue());
        neuralNetworkLSTM.setCurrentSquaredError(0.0017);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.001, neuralNetworkLSTM.getLearningStepValue());
        neuralNetworkLSTM.setCurrentSquaredError(0.00099);
        neuralNetworkLSTM.learningStepValueUpdate();
        Assertions.assertEquals(0.0001, neuralNetworkLSTM.getLearningStepValue());
    }

    private double weightGenerate() {
        this.weight = this.weight + WEIGHT_STEP_INCREMENT;
        if (this.weight > 1.0) {
            this.weight = WEIGHT_START_VALUE;
        }
        return this.weight;
    }

}