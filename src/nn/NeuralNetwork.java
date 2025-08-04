package nn;

import exceptions.NNInputExceptions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {
    List<Layer> layers;
    private int layersCount;

    public NeuralNetwork(int[] layersCountArray) {
        this.layers = new ArrayList<Layer>();
        this.layersCount = layersCountArray.length;
        this.layers.add(new Layer(layersCountArray[0], layersCountArray[0], 1, 0));
        for (int layerIndex = 1; layerIndex < this.layersCount; layerIndex++) {
            this.layers.add(new Layer(layersCountArray[layerIndex-1], layersCountArray[layerIndex], 1, layerIndex));
        }
    }

    public List<Layer> getLayers(){
        return this.layers;
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

    public void forwardPass(){
        this.layers.get(CommonConstants.FIRST_LAYER).calculateLayerOutputs();
        for (int layerIndex = CommonConstants.SECOND_LAYER; layerIndex < this.layersCount; layerIndex++) {
            this.layers.get(layerIndex).setInputs(this.layers.get(layerIndex - 1).getLayerOutputs());
            this.layers.get(layerIndex).calculateLayerOutputs();
        }
    }
}
