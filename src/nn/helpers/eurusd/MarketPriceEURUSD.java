package nn.helpers.eurusd;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;
import nn.interfaces.MarketPrice;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class MarketPriceEURUSD implements MarketPrice {
    private final static int DATE_POS = 0;
    private final static int TIME_POS = 1;
    private final static int OPEN_POS = 2;
    private final static int HIGH_POS = 3;
    private final static int LOW_POS = 4;
    private final static int CLOSE_POS = 5;
    private final static int VOLUME_POS = 6;
    private final static double NORMALIZE_PRICE_COEF = 10.0;
    private final static double NORMALIZE_DATE_TIME_COEF = 100.0;
    private final static double NORMALIZE_VOLUME_COEF = 10000.0;
    private final double day_of_month;
    private final double day_of_week;
    private final double hour;
    private final double month;
    private Date date;
    private double open;
    private double high;
    private double low;
    private double close;
    private double volume;

    public MarketPriceEURUSD(String line) {
        line = line.replace(CommonConstants.DOUBLE_QUOTE, CommonConstants.EMPTY);
        String[] textArray = line.split(CommonConstants.TAB_SYMBOL);
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy.M.dd hh:mm:ss");
        try {
            this.open = Double.valueOf(textArray[OPEN_POS]);
            this.high = Double.valueOf(textArray[HIGH_POS]);
            this.low = Double.valueOf(textArray[LOW_POS]);
            this.close = Double.valueOf(textArray[CLOSE_POS]);
            this.volume = Double.valueOf(textArray[VOLUME_POS]);
            this.date = simpleDateFormat.parse(textArray[DATE_POS].concat(CommonConstants.WHITE_SPACE).concat(textArray[TIME_POS]));
        } catch (Exception e) {
            throw new NNInputExceptions(CommonConstants.INCORRECT_MERKET_PRICE_DATA);
        }
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(this.date);
        this.month = calendar.get(Calendar.MONTH);
        this.day_of_month = calendar.get(Calendar.DAY_OF_MONTH);
        this.day_of_week = calendar.get(Calendar.DAY_OF_WEEK);
        this.hour = calendar.get(Calendar.HOUR);
    }

    @Override
    public double[] getPricesFlatData() {
        return new double[]{this.getDay_of_month(), this.getDay_of_week(), this.getHour(), this.getMonth(), this.getOpen(), this.getHigh(), this.getLow(), this.getClose(), this.getVolume()};
    }

    @Override
    public double[] getNormalizedPricesFlatData() {
        return new double[]{
                this.getDay_of_month() / NORMALIZE_DATE_TIME_COEF,
                this.getDay_of_week() / NORMALIZE_DATE_TIME_COEF,
                this.getHour() / NORMALIZE_DATE_TIME_COEF,
                this.getMonth() / NORMALIZE_DATE_TIME_COEF,
                this.getOpen() / NORMALIZE_PRICE_COEF,
                this.getHigh() / NORMALIZE_PRICE_COEF,
                this.getLow() / NORMALIZE_PRICE_COEF,
                this.getClose() / NORMALIZE_PRICE_COEF,
                this.getVolume() / NORMALIZE_VOLUME_COEF
        };
    }

    public Date getDate() {
        return this.date;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public double getOpen() {
        return open;
    }

    public void setOpen(double open) {
        this.open = open;
    }

    public double getHigh() {
        return high;
    }

    public void setHigh(double high) {
        this.high = high;
    }

    public double getLow() {
        return low;
    }

    public void setLow(double low) {
        this.low = low;
    }

    public double getClose() {
        return this.close;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public double getVolume() {
        return this.volume;
    }

    public void setVolume(double volume) {
        this.volume = volume;
    }

    public double getMonth() {
        return this.month;
    }

    public double getDay_of_month() {
        return this.day_of_month;
    }

    public double getDay_of_week() {
        return this.day_of_week;
    }

    public double getHour() {
        return this.hour;
    }
}
