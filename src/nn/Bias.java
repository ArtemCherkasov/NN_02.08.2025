package nn;

public class Bias {
    private double value;

    public Bias(){
        this.value = CommonConstants.BIAS_DEFAULT_VALUE;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
