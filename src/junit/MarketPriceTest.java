package junit;

import nn.helpers.MarketPriceHelper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

class MarketPriceTest {
    public final static double OPEN = 1.40954;
    public final static double HIGH = 1.41129;
    public final static double LOW = 1.40929;
    public final static double CLOSE = 1.41069;
    public final static double VOLUME = 1002;
    public final static String DATE_TIME = "2009.06.12 10:15:00";
    public final static int MONTH = 5;
    public final static int MONTH_DAY = 12;
    public final static String PRICE_TEXT = "2009.06.12\t10:15:00\t1.40954\t1.41129\t1.40929\t1.41069\t1002\t0\t20";

    MarketPriceHelper marketPrice;
    SimpleDateFormat dateTimeFormat;
    Date dateTime;

    @BeforeEach
    public void initNetwork() {
        marketPrice = new MarketPriceHelper(PRICE_TEXT);
        dateTimeFormat = new SimpleDateFormat("yyyy.M.dd hh:mm:ss");
        try {
            dateTime = dateTimeFormat.parse(DATE_TIME);

        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void networkConfigurationTest() {
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