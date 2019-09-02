package dataProcess;

import java.io.File;
import java.util.Date;

public class Main {

	public static void main(String[] args) {
		String file1 = "";
		String file2 = "";
		String input1 = "data/*.csv";
		String input2 = "data/*.csv";
		int stuSize = 5000;
		String output1 = "dataset/" + file1 + "Result.csv";
		String output2 = "dataset/" + file2 + "Result.csv";

		String begin = "2018-02-28 00:00:00";
		String end = "2018-12-09 00:00:00";
		int stepHours = 6;
		// 处理为SCALE输入文件
		Process process = new Process();
		process.init(new File(input1), begin, end, stepHours, stuSize);
		process.process(new File(output1));
		Process process1 = new Process();
		process1.init(new File(input2), begin, end, stepHours, stuSize);
		process1.process(new File(output2));
	}

}
