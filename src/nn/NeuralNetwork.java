package nn;

import exceptions.NNInputExceptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    List<Layer> layers;
    private int layersCount;
    private double learningRate;

    public NeuralNetwork(int[] layersCountArray) {
        this.layers = new ArrayList<Layer>();
        this.layersCount = layersCountArray.length;
        this.layers.add(new Layer(layersCountArray[0], layersCountArray[0], 1, 0));
        for (int layerIndex = 1; layerIndex < this.layersCount; layerIndex++) {
            this.layers.add(new Layer(layersCountArray[layerIndex-1], layersCountArray[layerIndex], 1, layerIndex));
        }
        this.learningRate = CommonConstants.LEARNING_RATE_DEFAULT_VALUE;
    }

    public List<Layer> getLayers(){
        return this.layers;
    }

    public Layer getLayer(int indexLayer){
        return this.layers.get(indexLayer);
    }

    public Layer getFirstLayer(){
        return this.layers.get(CommonConstants.FIRST_LAYER);
    }

    public Layer getLastLayer(){
        return this.layers.get(this.layers.size() - 1);
    }

    public String neuralNetworkShortInfo(){
        String result = CommonConstants.EMPTY;
        for(Layer layer: this.layers){
            result = result.concat(layer.layerInfo()).concat(System.lineSeparator());
        }
        return result;
    }

    public String getNetworkOutputInfo(){
        return Arrays.toString(getLastLayer().getLayerOutputs());
    }

    public void setInputsToNet(double[] inputs){
        if (inputs.length != getFirstLayer().getInputCount()){
            throw new NNInputExceptions(CommonConstants.INCORRECT_INPUTS_COUNT);
        }
        this.layers.get(CommonConstants.FIRST_LAYER).setInputs(inputs);
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void forwardPropagation(){
        this.layers.get(CommonConstants.FIRST_LAYER).calculateLayerOutputs();
        for (int layerIndex = CommonConstants.SECOND_LAYER; layerIndex < this.layersCount; layerIndex++) {
            this.layers.get(layerIndex).setInputs(this.layers.get(layerIndex - 1).getLayerOutputs());
            this.layers.get(layerIndex).calculateLayerOutputs();
        }
    }

    public double getErrorTotal(double[] target){
        double error = 0.0;
        for(int outputIndex = 0; outputIndex < this.getLastLayer().getLayerOutputs().length; outputIndex++){
            error = error + Math.pow(target[outputIndex] - this.getLastLayer().getLayerOutputs()[outputIndex],2.0) / 2.0;
        }
        return error;
    }

    public void calculateWeightDeltaLastLayer(double[] target){
        for(int outputIndex = 0; outputIndex < this.getLastLayer().getLayerOutputs().length; outputIndex++){
            double dEdOut = -1.0 * (target[outputIndex] - this.getLastLayer().getNode(outputIndex).getOutput());
            double dOutDNet = this.getLastLayer().getNode(outputIndex).getOutput()*(1 - this.getLastLayer().getNode(outputIndex).getOutput());
            this.getLastLayer().getNode(outputIndex).setDeltaOfNode(dEdOut*dOutDNet);
            double[] deltaOfWeight = new double[this.getLastLayer().getInputAndBiasesCount()];
            for (int inputIndex = 0; inputIndex < this.getLastLayer().getInputAndBiasesCount(); inputIndex++){
                deltaOfWeight[inputIndex] = this.learningRate*this.getLastLayer().getNode(outputIndex).getDeltaOfNode()*this.getLastLayer().getNode(outputIndex).getInputs()[inputIndex];
            }
            this.getLastLayer().getNode(outputIndex).setDeltaOfWeight(deltaOfWeight);
        }
    }

    public void weightUpdate(){
        for (int layerIndex = 0; layerIndex < this.layersCount; layerIndex++){
            for (int nodeIndex = 0; nodeIndex < this.getLayer(layerIndex).getNodesCount(); nodeIndex++){
                this.getLayer(layerIndex).getNode(nodeIndex).weightUpdate();
            }
        }
    }
}
