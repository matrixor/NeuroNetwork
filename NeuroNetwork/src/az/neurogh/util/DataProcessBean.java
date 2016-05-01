package az.neurogh.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataProcessBean {
	
	public static List<String> loadMaxBinaryData() {
		BufferedReader br = null;
		List<String> dataRows = new ArrayList<String>();
		try {
			br = new BufferedReader(new FileReader(Config.LOTTERY_MAX_DATA_FILE_PATH + Config.LOTTERY_MAX_BIN_DATA_FILE_NAME));
			String inputLine;
			while ((inputLine = br.readLine()) != null) {
				dataRows.add(inputLine);
			}	
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
    	     try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
		
		return dataRows;
	}

}
