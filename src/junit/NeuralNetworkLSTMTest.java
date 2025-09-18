package junit;

import nn.lstm.LSTMRow;
import nn.lstm.NeuralNetworkLSTM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuralNetworkLSTMTest {
    public final static int FIRST_CELL_NODES_COUNT = 3;
    public final static int SECOND_CELL_NODES_COUNT = 4;
    public final static int THIRD_CELL_NODES_COUNT = 3;

    LSTMRow lstmRow;
    NeuralNetworkLSTM nnLSTM;

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
        lstmRow.getCell(0).getInputGate().getNode(0).setCustomWeights(new double[]{0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075});
        lstmRow.getCell(0).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.080, 0.085, 0.090, 0.095, 0.100, 0.105, 0.110});
        lstmRow.getCell(0).getOutputGate().getNode(0).setCustomWeights(new double[]{0.115, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145});
        lstmRow.getCell(0).getForgetGate().getNode(1).setCustomWeights(new double[]{0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180});
        lstmRow.getCell(0).getInputGate().getNode(1).setCustomWeights(new double[]{0.185, 0.190, 0.195, 0.200, 0.205, 0.210, 0.215});
        lstmRow.getCell(0).getCandidateCellState().getNode(1).setCustomWeights(new double[]{0.220, 0.225, 0.230, 0.235, 0.240, 0.245, 0.250});
        lstmRow.getCell(0).getOutputGate().getNode(1).setCustomWeights(new double[]{0.255, 0.260, 0.265, 0.270, 0.275, 0.280, 0.285});
        lstmRow.getCell(0).getForgetGate().getNode(2).setCustomWeights(new double[]{0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320});
        lstmRow.getCell(0).getInputGate().getNode(2).setCustomWeights(new double[]{0.325, 0.330, 0.335, 0.340, 0.345, 0.350, 0.355});
        lstmRow.getCell(0).getCandidateCellState().getNode(2).setCustomWeights(new double[]{0.360, 0.365, 0.370, 0.375, 0.380, 0.385, 0.390});
        lstmRow.getCell(0).getOutputGate().getNode(2).setCustomWeights(new double[]{0.395, 0.400, 0.405, 0.410, 0.415, 0.420, 0.425});

        lstmRow.getCell(1).getForgetGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(1).getInputGate().getNode(0).setCustomWeights(new double[]{0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085});
        lstmRow.getCell(1).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125});
        lstmRow.getCell(1).getOutputGate().getNode(0).setCustomWeights(new double[]{0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165});
        lstmRow.getCell(1).getForgetGate().getNode(1).setCustomWeights(new double[]{0.170, 0.175, 0.180, 0.185, 0.190, 0.195, 0.200, 0.205});
        lstmRow.getCell(1).getInputGate().getNode(1).setCustomWeights(new double[]{0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240, 0.245});
        lstmRow.getCell(1).getCandidateCellState().getNode(1).setCustomWeights(new double[]{0.250, 0.255, 0.260, 0.265, 0.270, 0.275, 0.280, 0.285});
        lstmRow.getCell(1).getOutputGate().getNode(1).setCustomWeights(new double[]{0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.325});
        lstmRow.getCell(1).getForgetGate().getNode(2).setCustomWeights(new double[]{0.330, 0.335, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365});
        lstmRow.getCell(1).getInputGate().getNode(2).setCustomWeights(new double[]{0.370, 0.375, 0.380, 0.385, 0.390, 0.395, 0.400, 0.405});
        lstmRow.getCell(1).getCandidateCellState().getNode(2).setCustomWeights(new double[]{0.410, 0.415, 0.420, 0.425, 0.430, 0.435, 0.440, 0.445});
        lstmRow.getCell(1).getOutputGate().getNode(2).setCustomWeights(new double[]{0.450, 0.455, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485});
        lstmRow.getCell(1).getForgetGate().getNode(3).setCustomWeights(new double[]{0.490, 0.495, 0.500, 0.505, 0.510, 0.515, 0.520, 0.525});
        lstmRow.getCell(1).getInputGate().getNode(3).setCustomWeights(new double[]{0.530, 0.535, 0.540, 0.545, 0.550, 0.555, 0.560, 0.565});
        lstmRow.getCell(1).getCandidateCellState().getNode(3).setCustomWeights(new double[]{0.570, 0.575, 0.580, 0.585, 0.590, 0.595, 0.600, 0.605});
        lstmRow.getCell(1).getOutputGate().getNode(3).setCustomWeights(new double[]{0.610, 0.615, 0.620, 0.625, 0.630, 0.635, 0.640, 0.645});

        lstmRow.getCell(2).getForgetGate().getNode(0).setCustomWeights(new double[]{0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045});
        lstmRow.getCell(2).getInputGate().getNode(0).setCustomWeights(new double[]{0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085});
        lstmRow.getCell(2).getCandidateCellState().getNode(0).setCustomWeights(new double[]{0.090, 0.095, 0.100, 0.105, 0.110, 0.115, 0.120, 0.125});
        lstmRow.getCell(2).getOutputGate().getNode(0).setCustomWeights(new double[]{0.130, 0.135, 0.140, 0.145, 0.150, 0.155, 0.160, 0.165});
        lstmRow.getCell(2).getForgetGate().getNode(1).setCustomWeights(new double[]{0.170, 0.175, 0.180, 0.185, 0.190, 0.195, 0.200, 0.205});
        lstmRow.getCell(2).getInputGate().getNode(1).setCustomWeights(new double[]{0.210, 0.215, 0.220, 0.225, 0.230, 0.235, 0.240, 0.245});
        lstmRow.getCell(2).getCandidateCellState().getNode(1).setCustomWeights(new double[]{0.250, 0.255, 0.260, 0.265, 0.270, 0.275, 0.280, 0.285});
        lstmRow.getCell(2).getOutputGate().getNode(1).setCustomWeights(new double[]{0.290, 0.295, 0.300, 0.305, 0.310, 0.315, 0.320, 0.325});
        lstmRow.getCell(2).getForgetGate().getNode(2).setCustomWeights(new double[]{0.330, 0.335, 0.340, 0.345, 0.350, 0.355, 0.360, 0.365});
        lstmRow.getCell(2).getInputGate().getNode(2).setCustomWeights(new double[]{0.370, 0.375, 0.380, 0.385, 0.390, 0.395, 0.400, 0.405});
        lstmRow.getCell(2).getCandidateCellState().getNode(2).setCustomWeights(new double[]{0.410, 0.415, 0.420, 0.425, 0.430, 0.435, 0.440, 0.445});
        lstmRow.getCell(2).getOutputGate().getNode(2).setCustomWeights(new double[]{0.450, 0.455, 0.460, 0.465, 0.470, 0.475, 0.480, 0.485});

        nnLSTM = new NeuralNetworkLSTM(lstmRow);
        nnLSTM.addLSTMRowsSeries(30);
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
    void forwardPropogationLSRMRowTest() {
        lstmRow.forwardPropogationRow();
    }

}