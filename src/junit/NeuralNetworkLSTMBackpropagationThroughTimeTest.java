package junit;

import nn.lstm.LSTMRow;
import nn.lstm.NeuralNetworkLSTM;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuralNetworkLSTMBackpropagationThroughTimeTest {
    private final static int FIRST_CELL_NODES_COUNT = 2;

    LSTMRow lstmRow;
    NeuralNetworkLSTM nnLSTM;

    @BeforeEach
    public void initNetwork() {
        lstmRow = new LSTMRow(new int[]{FIRST_CELL_NODES_COUNT});
        lstmRow.setInputToLSTMRow(new double[]{0.05, 0.10});
        nnLSTM = new NeuralNetworkLSTM(lstmRow);
    }

    @Test
    void nnLSTMTest() {
        nnLSTM.forwardPropagationRow();
    }

}