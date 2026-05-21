package exceptions;

import nn.common.CommonConstants;

public class NNInputExceptions extends RuntimeException {
    public NNInputExceptions(String message) {
        super(message);
    }

    public NNInputExceptions(String message, Integer linePointer) {
        super(message.concat(CommonConstants.WHITE_SPACE).concat("(in line ").concat(linePointer.toString()).concat(")"));
    }

    public NNInputExceptions(String message, int current, int mustBe) {
        super(message.concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(current)).concat(CommonConstants.WHITE_SPACE).concat("must be").concat(CommonConstants.WHITE_SPACE).concat(String.valueOf(mustBe)));
    }
}
