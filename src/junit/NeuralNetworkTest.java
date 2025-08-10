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
        Assertions.assertEquals(FIRST_LAYER_NODES_COUNT, neuralNetwork.getLayer(0).getNodes().size());
        Assertions.assertEquals(SECOND_LAYER_NODES_COUNT, neuralNetwork.getLayer(1).getNodes().size());
    }

    @Test
    void inputLayersTest() {
        double[] expectedsInput = {FIRST_INPUT_TO_NETWORK, SECOND_INPUT_TO_NETWORK, CommonConstants.BIAS_DEFAULT_VALUE, FIRST_INPUT_TO_NETWORK, SECOND_INPUT_TO_NETWORK, CommonConstants.BIAS_DEFAULT_VALUE};
        Assertions.assertArrayEquals(expectedsInput, neuralNetwork.getFirstLayer().getFlatInputs(), 0.0);
    }

    @Test
    void calculationLayersTest(){
        neuralNetwork.getLayer(0).getNode(0).setCustomWeights(new double[]{0.15, 0.20, 0.35});
        neuralNetwork.getLayer(0).getNode(1).setCustomWeights(new double[]{0.25, 0.30, 0.35});
        neuralNetwork.getLayer(1).getNode(0).setCustomWeights(new double[]{0.40, 0.45, 0.60});
        neuralNetwork.getLayer(1).getNode(1).setCustomWeights(new double[]{0.50, 0.55, 0.60});
        neuralNetwork.forwardPropagation();
        double l0n0 = neuralNetwork.getLayer(0).getNode(0).getOutput();
        double l0n1 = neuralNetwork.getLayer(0).getNode(1).getOutput();
        double l1n0 = neuralNetwork.getLayer(1).getNode(0).getOutput();
        double l1n1 = neuralNetwork.getLayer(1).getNode(1).getOutput();
        Assertions.assertEquals(BigDecimal.valueOf(0.593269992), BigDecimal.valueOf(l0n0).setScale(9, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.596884378), BigDecimal.valueOf(l0n1).setScale(9, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.75136507), BigDecimal.valueOf(l1n0).setScale(8, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.772928465), BigDecimal.valueOf(l1n1).setScale(9, RoundingMode.HALF_UP));
        double error = neuralNetwork.getErrorTotal(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        Assertions.assertEquals(BigDecimal.valueOf(0.298371109), BigDecimal.valueOf(error).setScale(9, RoundingMode.HALF_UP));
        neuralNetwork.setLearningRate(CommonConstants.LEARNING_RATE_DEFAULT_VALUE);
        neuralNetwork.calculateWeightDeltaLastLayer(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        neuralNetwork.calculateWeightDeltaHiddenLayers();
        neuralNetwork.weightUpdate();
        double w0l0n0 = neuralNetwork.getLayer(1).getNode(1).getWeights()[0];
        double w1l0n0 = neuralNetwork.getLayer(1).getNode(1).getWeights()[1];
        double w2l0n0 = neuralNetwork.getLayer(1).getNode(1).getWeights()[2];
        Assertions.assertEquals(BigDecimal.valueOf(0.51130127), BigDecimal.valueOf(w0l0n0).setScale(8, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.561370121), BigDecimal.valueOf(w1l0n0).setScale(9, RoundingMode.HALF_UP));
        Assertions.assertEquals(BigDecimal.valueOf(0.619049118), BigDecimal.valueOf(w2l0n0).setScale(9, RoundingMode.HALF_UP));
        neuralNetwork.getLayer(0).getNode(0).setCustomWeight(2, 0.35);
        neuralNetwork.getLayer(0).getNode(1).setCustomWeight(2, 0.35);
        neuralNetwork.getLayer(1).getNode(0).setCustomWeight(2, 0.60);
        neuralNetwork.getLayer(1).getNode(1).setCustomWeight(2, 0.60);
        neuralNetwork.forwardPropagation();
        error = neuralNetwork.getErrorTotal(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        Assertions.assertEquals(BigDecimal.valueOf(0.291027774), BigDecimal.valueOf(error).setScale(9, RoundingMode.HALF_UP));
        for (int i = 1; i < 12000; i++){
            neuralNetwork.calculateWeightDeltaLastLayer(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
            neuralNetwork.calculateWeightDeltaHiddenLayers();
            neuralNetwork.weightUpdate();
            neuralNetwork.forwardPropagation();
        }
        error = neuralNetwork.getErrorTotal(new double[]{FIRST_OUTPUT_FROM_NETWORK, SECOND_OUTPUT_FROM_NETWORK});
        Assertions.assertEquals(BigDecimal.valueOf(0.000001309), BigDecimal.valueOf(error).setScale(9, RoundingMode.HALF_UP));
    }
}