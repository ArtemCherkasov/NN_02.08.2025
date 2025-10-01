package nn.helpers;

import nn.helpers.eurusd.MarketPriceEURUSD;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class DataHelper {
    private static List<String> priceLines;
    private static List<MarketPriceEURUSD> marketPrices;

    public static List<String> loadTextDataFromFile(String pathToFile) throws IOException {
        return Files.lines(Paths.get(pathToFile), StandardCharsets.UTF_8).toList();
    }

    public static List<MarketPriceEURUSD> loadMarketPricesFromFile(String pathToFile) {
        try {
            priceLines = DataHelper.loadTextDataFromFile(pathToFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        marketPrices = new ArrayList<MarketPriceEURUSD>();
        for (String line : priceLines) {
            try {
                marketPrices.add(new MarketPriceEURUSD(line));
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
}
