package junit;

import nn.CommonConstants;
import nn.NeuralNetwork;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.RoundingMode;

class NeuralNetworkTest {
    public final static int FIRST_LAYER_NODES_COUNT = 2;
    public final static int SECOND_LAYER_NODES_COUNT = 2;
    public final static double FIRST_INPUT_TO_NETWORK = 0.05;
    public final static double SECOND_INPUT_TO_NETWORK = 0.1;
    public final static double FIRST_OUTPUT_FROM_NETWORK = 0.01;
    public final static double SECOND_OUTPUT_FROM_NETWORK = 0.99;

    NeuralNetwork neuralNetwork;

    @BeforeEach
    public void initNetwork() {
        neuralNetwork = new NeuralNetwork(new int[]{FIRST_LAYER_NODES_COUNT, SECOND_LAYER_NODES_COUNT});
        neuralNetwork.setInputsToNet(new double[]{FIRST_INPUT_TO_NETWORK, SECOND_INPUT_TO_NETWORK});
    }

    @Test
    void networkConfigurationTest() {
        Assertions.assertEquals(2, neuralNetwork.getLayers().size());
        Assertions.assertEquals(FIRST_LAYER_NODES_COUNT, neuralNetwork.getLayers().get(0).getNodes().size());
        Assertions.assertEquals(SECOND_LAYER_NODES_COUNT, neuralNetwork.getLayers().get(1).getNodes().size());
    }

    @Test
    void inputLayersTest() {
        double[] expectedsInput = {FIRST_INPUT_TO_NETWORK, SECOND_INPUT_TO_NETWORK, CommonConstants.BIAS_DEFAULT_VALUE, FIRST_INPUT_TO_NETWORK, SECOND_INPUT_TO_NETWORK, CommonConstants.BIAS_DEFAULT_VALUE};
        Assertions.assertArrayEquals(expectedsInput, neuralNetwork.getFirstLayer().getFlatInputs(), 0.0);
    }

    @Test
    void calculationLayersTest(){
        neuralNetwork.getLayers().get(0).getNodes().get(0).setCustomWeights(new double[]{0.15, 0.20, 0.35});
        neuralNetwork.getLayers().get(0).getNodes().get(1).setCustomWeights(new double[]{0.25, 0.30, 0.35});
        neuralNetwork.getLayers().get(1).getNodes().get(0).setCustomWeights(new double[]{0.40, 0.45, 0.60});
        neuralNetwork.getLayers().get(1).getNodes().get(1).setCustomWeights(new double[]{0.50, 0.55, 0.60});
        neuralNetwork.forwardPropagation();
        double l0n0 = neuralNetwork.getLayers().get(0).getNodes().get(0).getOutput();
        double l0n1 = neuralNetwork.getLayers().get(0).getNodes().get(1).getOutput();
        double l1n0 = neuralNetwork.getLayers().get(1).getNodes().get(0).getOutput();
        double l1n1 = neuralNetwork.getLayers().get(1).getNodes().get(1).getOutput();
        Assertions.assertEquals(BigDecimal.valueOf(0.593269992), BigDecimal.valueOf(l0n0).setScale(9, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.596884378), BigDecimal.valueOf(l0n1).setScale(9, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.75136507), BigDecimal.valueOf(l1n0).setScale(8, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.772928465), BigDecimal.valueOf(l1n1).setScale(9, RoundingMode.HALF_UP));
        double error = neuralNetwork.getErrorTotal(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        Assertions.assertEquals(BigDecimal.valueOf(0.298371109), BigDecimal.valueOf(error).setScale(9, RoundingMode.HALF_UP));
        double[] errors = neuralNetwork.getDErrorTotalDOutput(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        Assertions.assertEquals(0, 0);
    }
}