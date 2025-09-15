import nn.simple.NeuralNetworkSimple;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {
    public static void main(String[] args) {
        NeuralNetworkSimple neuralNetwork = new NeuralNetworkSimple(new int[]{2, 2, 2});
        neuralNetwork.setInputsToNet(new double[]{0.5, 0.1});
        neuralNetwork.forwardPropagation();
        System.out.println(neuralNetwork.neuralNetworkShortInfo());
        System.out.println(neuralNetwork.getNetworkOutputInfo());
    }
}