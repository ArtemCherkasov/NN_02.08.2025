package nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class Layer {
    private List<Node> nodes;
    private List<Bias> biases;
    private int inputCount;
    private int nodesCount;
    private int biasesCount;
    private int layerIndex;
    private String layerName;

    public Layer(int inputCount, int nodesCount, int biasesCount, int layerIndex){
        this.inputCount = inputCount;
        this.nodesCount = nodesCount;
        this.nodes = new ArrayList<Node>();
        if (biasesCount > 0) {
            this.biasesCount = biasesCount;
            this.biases = new ArrayList<Bias>(biasesCount);
            for (int biasIndex = 0; biasIndex < biasesCount; biasIndex++){
                this.biases.add(new Bias());
            }
        }
        for (int i = 0; i < this.nodesCount; i++) {
            this.nodes.add(new Node(this.inputCount + this.biasesCount));
        }
        this.layerIndex = layerIndex;
        this.layerName = CommonConstants.LAYER_NAME_PREFIX.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(layerIndex));
    }

    public List<Node> getNodes() {
        return this.nodes;
    }

    public Node getNode(int nodeIndex) {
        return this.nodes.get(nodeIndex);
    }

    public void calculateLayerOutputs(){
        for (int i = 0; i < this.nodesCount; i++) {
            this.nodes.get(i).calculateOutput();
        }
    }

    public void setInputs(double[] inputs){
        for(Node node : this.nodes){
            double[] _inputs = DoubleStream.concat(Arrays.stream(inputs), this.biases.stream().mapToDouble(bias -> bias.getValue())).toArray();
            node.setInputs(_inputs);
        }
    }

    public double[] getLayerOutputs(){
        return this.nodes.stream().mapToDouble(node -> node.getOutput()).toArray();
    }

    public double[] getFlatInputs(){
        DoubleStream inputs = this.nodes.stream().flatMapToDouble(node -> DoubleStream.of(node.getInputs()));
        return inputs.toArray();
    }

    public double[] getLayerSumFunctions(){
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

    public String layerInfo() {
        String head = this.layerName;
        String nodesInfo = CommonConstants.LAYER_NODES_INFO_HEAD.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(this.nodesCount));
        String biasesInfo = CommonConstants.LAYER_BIASES_INFO_HEAD.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(this.biasesCount));
        return head.concat(System.lineSeparator()).concat(nodesInfo).concat(System.lineSeparator()).concat(biasesInfo);
    }
}
