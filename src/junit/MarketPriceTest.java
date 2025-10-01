package junit;

import nn.helpers.eurusd.MarketPriceEURUSD;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

class MarketPriceTest {
    private final static double OPEN = 1.40954;
    private final static double HIGH = 1.41129;
    private final static double LOW = 1.40929;
    private final static double CLOSE = 1.41069;
    private final static double VOLUME = 1002;
    private final static String DATE_TIME = "2009.06.12 10:15:00";
    private final static int MONTH = 5;
    private final static int MONTH_DAY = 12;
    private final static String PRICE_TEXT = "2009.06.12\t10:15:00\t1.40954\t1.41129\t1.40929\t1.41069\t1002\t0\t20";

    MarketPriceEURUSD marketPrice;
    SimpleDateFormat dateTimeFormat;
    Date dateTime;

    @BeforeEach
    public void initMarketPriceHelper() {
        marketPrice = new MarketPriceEURUSD(PRICE_TEXT);
        dateTimeFormat = new SimpleDateFormat("yyyy.M.dd hh:mm:ss");
        try {
            dateTime = dateTimeFormat.parse(DATE_TIME);

        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void marketPriceHelperTest() {
        Assertions.assertEquals(OPEN, marketPrice.getOpen());
        Assertions.assertEquals(HIGH, marketPrice.getHigh());
        Assertions.assertEquals(LOW, marketPrice.getLow());
        Assertions.assertEquals(CLOSE, marketPrice.getClose());
        Assertions.assertEquals(VOLUME, marketPrice.getVolume());
        Assertions.assertEquals(dateTime, marketPrice.getDate());
        Assertions.assertEquals(MONTH, marketPrice.getMonth());
        Assertions.assertEquals(MONTH_DAY, marketPrice.getDay_of_month());
    }
}