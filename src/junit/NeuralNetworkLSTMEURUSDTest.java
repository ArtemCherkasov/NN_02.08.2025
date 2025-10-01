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

class NeuralNetworkLSTMEURUSDTest {
    private final static int FIRST_CELL_NODES_COUNT = 9;
    private final static int SECOND_CELL_NODES_COUNT = 9;
    private final static int THIRD_CELL_NODES_COUNT = 5;
    private final static double WEIGHT_START_VALUE = 0.005;
    private final static double WEIGHT_STEP_INCREMENT = 0.005;
    private final static String PATH_TO_DATA_DIR = "\\resources\\";
    private final static String FILE_NAME = "\\EURUSD_H1_200906120000_202509251100.csv\\";

    LSTMRow lstmRow;
    NeuralNetworkLSTM nnLSTM;
    List<MarketPriceEURUSD> marketPrices;
    double weight = WEIGHT_START_VALUE;

    @BeforeEach
    public void initNetwork() {
        lstmRow = new LSTMRow(new int[]{FIRST_CELL_NODES_COUNT, SECOND_CELL_NODES_COUNT, THIRD_CELL_NODES_COUNT});
        nnLSTM = new NeuralNetworkLSTM(lstmRow);
        nnLSTM.addLSTMRowsSeries(29);
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
        nnLSTM.setInputSeries(DataHelper.getNormalMarketPriseSeriesFromList(marketPrices, 0, 30));
        for (LSTMRow row : nnLSTM.getLstmRowList()) {
            for (LSTMCell cell : row.getCellList()) {
                for (Node node : cell.getInputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, getAutoWeight());
                    }
                }
                for (Node node : cell.getOutputGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, getAutoWeight());
                    }
                }
                for (Node node : cell.getForgetGate().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, getAutoWeight());
                    }
                }
                for (Node node : cell.getCandidateCellState().getNodes()) {
                    for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                        node.setCustomWeight(weightIndex, getAutoWeight());
                    }
                }
            }
            for (Node node : row.getLastLayer().getNodes()) {
                for (int weightIndex = 0; weightIndex < node.getWeights().length; weightIndex++) {
                    node.setCustomWeight(weightIndex, getAutoWeight());
                }
            }
        }
    }

    @Test
    void lstmRowConfigurationTest() {
        Assertions.assertEquals(3, lstmRow.getCellList().size());
        Assertions.assertEquals(FIRST_CELL_NODES_COUNT, lstmRow.getCell(0).getForgetGate().getNodes().size());
        Assertions.assertEquals(SECOND_CELL_NODES_COUNT, lstmRow.getCell(1).getForgetGate().getNodes().size());
        Assertions.assertEquals(THIRD_CELL_NODES_COUNT, lstmRow.getCell(2).getForgetGate().getNodes().size());
        Assertions.assertEquals(FIRST_CELL_NODES_COUNT, lstmRow.getCell(0).getInputGate().getNodes().size());
        Assertions.assertEquals(SECOND_CELL_NODES_COUNT, lstmRow.getCell(1).getInputGate().getNodes().size());
        Assertions.assertEquals(THIRD_CELL_NODES_COUNT, lstmRow.getCell(2).getInputGate().getNodes().size());
        Assertions.assertEquals(FIRST_CELL_NODES_COUNT, lstmRow.getCell(0).getOutputGate().getNodes().size());
        Assertions.assertEquals(SECOND_CELL_NODES_COUNT, lstmRow.getCell(1).getOutputGate().getNodes().size());
        Assertions.assertEquals(THIRD_CELL_NODES_COUNT, lstmRow.getCell(2).getOutputGate().getNodes().size());
        Assertions.assertEquals(FIRST_CELL_NODES_COUNT, lstmRow.getCell(0).getCandidateCellState().getNodes().size());
        Assertions.assertEquals(SECOND_CELL_NODES_COUNT, lstmRow.getCell(1).getCandidateCellState().getNodes().size());
        Assertions.assertEquals(THIRD_CELL_NODES_COUNT, lstmRow.getCell(2).getCandidateCellState().getNodes().size());
    }

    @Test
    void nnLSTMTest() {
        nnLSTM.forwardPropogationRow();
        Assertions.assertEquals(0.9380921371519626, nnLSTM.getLstmRowList().get(29).getLastLSTMCellOutput()[0]);
        Assertions.assertEquals(0.9380951995116282, nnLSTM.getLstmRowList().get(29).getLastLSTMCellOutput()[1]);
        Assertions.assertEquals(0.9380962646036628, nnLSTM.getLstmRowList().get(29).getLastLSTMCellOutput()[2]);
    }

    private double getAutoWeight() {
        this.weight = this.weight + WEIGHT_STEP_INCREMENT;
        if (this.weight > 1.0) {
            this.weight = WEIGHT_START_VALUE;
        }

        return this.weight;
    }

}