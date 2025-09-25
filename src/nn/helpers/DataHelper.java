package nn.helpers;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class DataHelper {
    private List<String> prices;

    public static List<String> loadDataFromFile(String pathToFile) throws IOException {
       return Files.lines(Paths.get(pathToFile), StandardCharsets.UTF_8).toList();
    }
}
