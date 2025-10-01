package nn.helpers.eurusd;

import java.util.List;

public class PriceSigmaConverter {
    private final static double EURUSD_H1_AVERAGE_DEVIATION = 0.00085;
    private final static double EURUSD_H1_SIGMA_0_75_X = 1.0986122886681098;
    private final static double EURUSD_H1_SIGMA_0_25_X = -1.0986122886681098;
    private final static double PRICE_CONVERTED_TO_SIGMA_X = EURUSD_H1_SIGMA_0_75_X / EURUSD_H1_AVERAGE_DEVIATION;
    private final static double SIGMA_IN_X0 = 0.5;
    private double maximumHeightDeviation;
    private double maximumLowDeviation;
    private double averageHeightDeviation = 0.0;
    private double averageLowDeviation = 0.0;
    private double averagePrice = 0.0;

    public PriceSigmaConverter() {
    }

    public void maxDeviation(List<MarketPriceEURUSD> marketPrices) {
        for (int i = 1; i < marketPrices.size(); i++) {
            averagePrice = averagePrice + marketPrices.get(i).getClose();
        }
        int n = 0;
        int m = 0;
        for (MarketPriceEURUSD mp : marketPrices) {
            if (0 < (mp.getHigh() - mp.getOpen())) {
                n++;
                averageHeightDeviation = averageHeightDeviation + (mp.getHigh() - mp.getClose());
            }
            if (0 < (mp.getOpen() - mp.getLow())) {
                m++;
                averageLowDeviation = averageLowDeviation + (mp.getClose() - mp.getLow());
            }
            if (maximumHeightDeviation < (mp.getHigh() - mp.getOpen())) {
                maximumHeightDeviation = mp.getHigh() - mp.getOpen();
            }
            if (maximumLowDeviation < (mp.getOpen() - mp.getLow())) {
                maximumLowDeviation = mp.getOpen() - mp.getLow();
            }
        }
        averageHeightDeviation = averageHeightDeviation / n;
        averageLowDeviation = averageLowDeviation / m;
        averagePrice = averagePrice / marketPrices.size();
    }

    public double inverseSigma(double y) {
        return Math.log(y / (1 - y));
    }

    public double deltaPriceConvertedToSigma(double deltaPrice) {
        return 1.0 / (1.0 + Math.exp(-1 * deltaPrice * PRICE_CONVERTED_TO_SIGMA_X));
    }

    public double sigmaToDeltaPrice(double sigma) {
        return inverseSigma(sigma) / PRICE_CONVERTED_TO_SIGMA_X;
    }
}
