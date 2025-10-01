package junit;

import nn.helpers.DataHelper;
import nn.helpers.eurusd.MarketPriceEURUSD;
import nn.helpers.eurusd.PriceSigmaConverter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

class DataHelperTest {
    private final static double OPEN = 1.08822;
    private final static double HIGH = 1.0885;
    private final static double LOW = 1.08756;
    private final static double CLOSE = 1.0883;
    private final static double VOLUME = 1626.0;
    private final static double MONTH = 7.0;
    private final static double MONTH_DAY = 18.0;
    private final static String PATH_TO_DATA_DIR = "\\resources\\";
    private final static String FILE_NAME = "\\EURUSD_H1_200906120000_202509251100.csv\\";
    private List<String> priceLines;
    private List<MarketPriceEURUSD> marketPrices;

    @BeforeEach
    public void loadData() {
        String filePath = System.getProperty("user.dir").concat(PATH_TO_DATA_DIR);
        marketPrices = DataHelper.loadMarketPricesFromFile(filePath.concat(FILE_NAME));
    }

    @Test
    void loadDataTest() {
        Assertions.assertArrayEquals(new double[]{12.0, 6.0, 0.0, 5.0, 1.40954, 1.41129, 1.40929, 1.41069, 1002.0}, marketPrices.get(0).getPricesFlatData());
        Assertions.assertEquals(OPEN, marketPrices.get(88000).getOpen());
        Assertions.assertEquals(HIGH, marketPrices.get(88000).getHigh());
        Assertions.assertEquals(LOW, marketPrices.get(88000).getLow());
        Assertions.assertEquals(CLOSE, marketPrices.get(88000).getClose());
        Assertions.assertEquals(VOLUME, marketPrices.get(88000).getVolume());
        Assertions.assertEquals(MONTH, marketPrices.get(88000).getMonth());
        Assertions.assertEquals(MONTH_DAY, marketPrices.get(88000).getDay_of_month());
        PriceSigmaConverter ps = new PriceSigmaConverter();
        ps.maxDeviation(marketPrices);
        Assertions.assertEquals(0.75, ps.deltaPriceConvertedToSigma(0.00085));
        Assertions.assertEquals(0.25, ps.deltaPriceConvertedToSigma(-0.00085));
        Assertions.assertEquals(0.00085, ps.sigmaToDeltaPrice(0.75));
        Assertions.assertEquals(-0.00085, ps.sigmaToDeltaPrice(0.25));
    }
}