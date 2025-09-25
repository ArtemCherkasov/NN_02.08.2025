package nn.helpers;

import exceptions.NNInputExceptions;
import nn.common.CommonConstants;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class MarketPrice {
    private final static int DATE_POS = 0;
    private final static int TIME_POS = 1;
    private final static int OPEN_POS = 2;
    private final static int HIGH_POS = 3;
    private final static int LOW_POS = 4;
    private final static int CLOSE_POS = 5;
    private final static int VOLUME_POS = 6;
    private Date date;
    private final int day_of_month;
    private final int day_of_week;
    private final int hour;
    private final int month;
    private double open;
    private double high;
    private double low;
    private double close;
    private double volume;

    public MarketPrice(String line) {
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
        month = calendar.get(Calendar.MONTH);
        day_of_month = calendar.get(Calendar.DAY_OF_MONTH);
        day_of_week = calendar.get(Calendar.DAY_OF_WEEK);
        hour = calendar.get(Calendar.HOUR);
    }

    public Date getDate() {
        return date;
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
        return close;
    }

    public void setClose(double close) {
        this.close = close;
    }

    public double getVolume() {
        return volume;
    }

    public void setVolume(double volume) {
        this.volume = volume;
    }

    public int getMonth() {
        return month;
    }

    public int getDay_of_month() {
        return day_of_month;
    }

    public int getDay_of_week() {
        return day_of_week;
    }

    public int getHour() {
        return hour;
    }
}
