package nn.helpers;

import nn.common.Node;
import nn.helpers.eurusd.MarketPriceEURUSD;
import nn.lstm.LSTMCell;
import nn.lstm.LSTMRow;
import nn.lstm.NeuralNetworkLSTM;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataHelper {
    private static List<String> priceLines;
    private static List<MarketPriceEURUSD> marketPrices;

    public static List<String> loadTextDataFromFile(String pathToFile) throws IOException {
        return Files.lines(Paths.get(pathToFile), StandardCharsets.UTF_8).toList();
    }

    public static List<MarketPriceEURUSD> loadMarketPricesFromFile(String pathToFile) {
        int linesCount = 0;
        try {
            priceLines = DataHelper.loadTextDataFromFile(pathToFile);
            linesCount = priceLines.size();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        marketPrices = new ArrayList<MarketPriceEURUSD>();
        for (int linePointer = 0; linePointer < linesCount; ++linePointer) {
            try {
                marketPrices.add(new MarketPriceEURUSD(priceLines.get(linePointer), linePointer));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return marketPrices;
    }

    public static double[][] getMarketPriseSeriesFromList(List<MarketPriceEURUSD> marketPricesList, int from, int to) {
        double[][] series = new double[to - from][marketPricesList.size()];
        int seriesIndex = 0;
        for (int marketPriceIndex = from; marketPriceIndex < to; marketPriceIndex++) {
            series[seriesIndex] = marketPricesList.get(marketPriceIndex).getPricesFlatData();
            seriesIndex++;
        }
        return series;
    }

    public static double[][] getNormalMarketPriseSeriesFromList(List<MarketPriceEURUSD> marketPricesList, int from, int to) {
        double[][] series = new double[to - from][marketPricesList.size()];
        int seriesIndex = 0;
        for (int marketPriceIndex = from; marketPriceIndex < to; marketPriceIndex++) {
            series[seriesIndex] = marketPricesList.get(marketPriceIndex).getNormalizedPricesFlatData();
            seriesIndex++;
        }
        return series;
    }

    public static void saveLSTMData(String pathToFile, NeuralNetworkLSTM neuralNetwork) throws IOException {
        Files.deleteIfExists(Paths.get(pathToFile));
        Files.createFile(Paths.get(pathToFile));
        for (LSTMRow row : neuralNetwork.getLstmRowList()) {
            for (LSTMCell cell : row.getCellList()) {
                for (Node node : cell.getInputGate().getNodes()) {
                    List<String> line = Arrays.asList(Arrays.toString(node.getWeights()));
                    Files.write(Paths.get(pathToFile), line, StandardOpenOption.APPEND);
                }
                for (Node node : cell.getOutputGate().getNodes()) {
                    List<String> line = Arrays.asList(Arrays.toString(node.getWeights()));
                    Files.write(Paths.get(pathToFile), line, StandardOpenOption.APPEND);
                }
                for (Node node : cell.getForgetGate().getNodes()) {
                    List<String> line = Arrays.asList(Arrays.toString(node.getWeights()));
                    Files.write(Paths.get(pathToFile), line, StandardOpenOption.APPEND);
                }
                for (Node node : cell.getCandidateCellState().getNodes()) {
                    List<String> line = Arrays.asList(Arrays.toString(node.getWeights()));
                    Files.write(Paths.get(pathToFile), line, StandardOpenOption.APPEND);
                }
            }
        }
    }

    public static void loadLSTMData(String pathToFile, NeuralNetworkLSTM neuralNetwork) throws IOException {
        List<String> lines = Files.lines(Paths.get(pathToFile)).toList();
        int lineIndex = 0;
        for (LSTMRow row : neuralNetwork.getLstmRowList()) {
            for (LSTMCell cell : row.getCellList()) {
                for (Node node : cell.getInputGate().getNodes()) {
                    double[] weightDataArray = Arrays.stream(lines.get(lineIndex).replace("[", "").replace("]", "").split(", ")).mapToDouble(Double::parseDouble).toArray();
                    node.setCustomWeights(weightDataArray);
                    lineIndex++;
                }
                for (Node node : cell.getOutputGate().getNodes()) {
                    double[] weightDataArray = Arrays.stream(lines.get(lineIndex).replace("[", "").replace("]", "").split(", ")).mapToDouble(Double::parseDouble).toArray();
                    node.setCustomWeights(weightDataArray);
                    lineIndex++;
                }
                for (Node node : cell.getForgetGate().getNodes()) {
                    double[] weightDataArray = Arrays.stream(lines.get(lineIndex).replace("[", "").replace("]", "").split(", ")).mapToDouble(Double::parseDouble).toArray();
                    node.setCustomWeights(weightDataArray);
                    lineIndex++;
                }
                for (Node node : cell.getCandidateCellState().getNodes()) {
                    double[] weightDataArray = Arrays.stream(lines.get(lineIndex).replace("[", "").replace("]", "").split(", ")).mapToDouble(Double::parseDouble).toArray();
                    node.setCustomWeights(weightDataArray);
                    lineIndex++;
                }
            }
        }
    }
}
