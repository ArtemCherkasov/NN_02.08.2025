package junit;

import nn.lstm.LSTMRow;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuralNetworkLSTMTest {
    public final static int FIRST_CELL_NODES_COUNT = 3;
    public final static int SECOND_CELL_NODES_COUNT = 4;
    public final static int THIRD_CELL_NODES_COUNT = 3;
    public final static double FIRST_INPUT_TO_NETWORK = 0.05;
    public final static double SECOND_INPUT_TO_NETWORK = 0.1;
    public final static double FIRST_OUTPUT_FROM_NETWORK = 0.01;
    public final static double SECOND_OUTPUT_FROM_NETWORK = 0.99;

    LSTMRow lstmRow;

    @BeforeEach
    public void initNetwork() {
        lstmRow = new LSTMRow(new int[]{FIRST_CELL_NODES_COUNT, SECOND_CELL_NODES_COUNT, THIRD_CELL_NODES_COUNT});
        lstmRow.setInputToLSTMRow(new double[]{0.05, 0.10, 0.15});
        lstmRow.getCell(0).setCellStateInput(new double[]{0.001, 0.001, 0.001});
        lstmRow.getCell(1).setCellStateInput(new double[]{0.001, 0.001, 0.001, 0.001});
        lstmRow.getCell(2).setCellStateInput(new double[]{0.001, 0.001, 0.001});
        lstmRow.getCell(0).setHiddenStateInput(new double[]{0.001, 0.001, 0.001});
        lstmRow.getCell(1).setHiddenStateInput(new double[]{0.001, 0.001, 0.001, 0.001});
        lstmRow.getCell(2).setHiddenStateInput(new double[]{0.001, 0.001, 0.001});

        lstmRow.getCell(0).getForgetGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getInputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getOutputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getForgetGate().getNode(1).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getInputGate().getNode(1).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getCandidateCellState().getNode(1).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});
        lstmRow.getCell(0).getOutputGate().getNode(1).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040});

        lstmRow.getCell(1).getForgetGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(1).getInputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(1).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(1).getOutputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});

        lstmRow.getCell(2).getForgetGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(2).getInputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(2).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(2).getOutputGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
    }

    @Test
    void networkConfigurationTest() {
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

}