package nn.common;

import nn.interfaces.LayerInterface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;

public class Layer implements LayerInterface {
    private final List<Node> nodes;
    private List<Bias> biases;
    private final int inputCount;
    private final int nodesCount;
    private int biasesCount;
    private final int layerIndex;
    private final String layerName;

    public Layer(int inputCount, int nodesCount, int biasesCount, int layerIndex) {
        this.inputCount = inputCount;
        this.nodesCount = nodesCount;
        this.nodes = new ArrayList<Node>();
        if (biasesCount > 0) {
            this.biasesCount = biasesCount;
            this.biases = new ArrayList<Bias>(biasesCount);
            for (int biasIndex = 0; biasIndex < biasesCount; biasIndex++) {
                this.biases.add(new Bias());
            }
        }
        for (int i = 0; i < this.nodesCount; i++) {
            this.nodes.add(new Node(this.inputCount + this.biasesCount));
        }
        this.layerIndex = layerIndex;
        this.layerName = CommonConstants.LAYER_NAME_PREFIX.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(layerIndex));
    }

    public Layer(Layer layer) {
        this.inputCount = layer.inputCount;
        this.nodesCount = layer.nodesCount;
        this.biasesCount = layer.biasesCount;
        this.layerIndex = layer.layerIndex;
        this.layerName = layer.layerName;
        this.nodes = new ArrayList<Node>();
        this.biases = new ArrayList<Bias>();
        for (Node node : layer.nodes) {
            this.nodes.add(new Node(node));
        }
        for (Bias bias : layer.biases) {
            this.biases.add(new Bias(bias));
        }
    }

    public List<Node> getNodes() {
        return this.nodes;
    }

    public Node getNode(int nodeIndex) {
        return this.nodes.get(nodeIndex);
    }

    public void calculateLayerSigmaOutputs() {
        for (int i = 0; i < this.nodesCount; i++) {
            this.nodes.get(i).calculateSigmaOutput();
        }
    }

    public void calculateLayerTanhOutputs() {
        for (int i = 0; i < this.nodesCount; i++) {
            this.nodes.get(i).calculateHiberbolicTangentOutput();
        }
    }

    public void setInputs(double[] inputs) {
        for (Node node : this.nodes) {
            double[] _inputs = DoubleStream.concat(Arrays.stream(inputs), this.biases.stream().mapToDouble(bias -> bias.getValue())).toArray();
            node.setInputs(_inputs);
        }
    }

    public double[] getLayerOutputs() {
        return this.nodes.stream().mapToDouble(node -> node.getNodeValue()).toArray();
    }

    public double[] getFlatInputs() {
        DoubleStream inputs = this.nodes.stream().flatMapToDouble(node -> DoubleStream.of(node.getInputs()));
        return inputs.toArray();
    }

    public double[] getLayerSumFunctions() {
        return this.nodes.stream().mapToDouble(node -> node.getSum()).toArray();
    }

    public List<Bias> getBiases() {
        return biases;
    }

    public int getInputCount() {
        return inputCount;
    }

    public int getBiasesCount() {
        return biasesCount;
    }

    public int getNodesCount() {
        return nodesCount;
    }

    public int getInputAndBiasesCount() {
        return this.inputCount + this.biasesCount;
    }

    public double getErrorFromAllNodes(int weightIndex) {
        double result = 0.0;
        for (int nodeIndex = 0; nodeIndex < this.nodesCount; nodeIndex++) {
            double dE_dOutHidden = this.getNode(nodeIndex).getDeltaOfNode() * this.getNode(nodeIndex).getWeight(weightIndex);
            result = result + dE_dOutHidden;
        }
        return result;
    }

    public String layerInfo() {
        String head = this.layerName;
        String nodesInfo = CommonConstants.LAYER_NODES_INFO_HEAD.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(this.nodesCount));
        String biasesInfo = CommonConstants.LAYER_BIASES_INFO_HEAD.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(this.biasesCount));
        return head.concat(System.lineSeparator()).concat(nodesInfo).concat(System.lineSeparator()).concat(biasesInfo);
    }
}
