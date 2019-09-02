package coVisitation;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Set;

import MRR.MRR;
import Utils.Utils;

public class Main {

	public static void main(String[] args) {
		String path = "dataset/*";
		File filePath = new File(path);
		File[] fileList = filePath.listFiles();
		List<File> embFile = new ArrayList<>();
		String resultFile = "*";
		String coFile1 = "*";
		String coFile2 = "*";
		String coOutFile = "*";
		String begin = "2018-02-28 00:00:00";
		String end = "2018-12-09 00:00:00";
		int seconds = 120; // 共现的时间

		int mrrInterval = 500; // 用于MRR测试的学生个数
		Utils u = new Utils();
		Date b = u.converTime(begin);
		Date e = u.converTime(end);

		long tStart = System.currentTimeMillis();

		// 初始化CoVisit
		CoVisit cv = new CoVisit(new File(coFile1), new File(coFile2), b, e);
		long iEnd = System.currentTimeMillis();
		System.out.println("初始化共运行" + (iEnd - tStart) / 1000 + "秒");

		// 获得计算的CoVisit结果
		Map<String, Map<String, Integer>> coVisit = cv.getCoVisit(seconds);
		Set<String> s = cv.getStudents();
		long cEnd = System.currentTimeMillis();
		System.out.println("co-visitation共运行" + (cEnd - iEnd) / 1000 + "秒");
		try {
			cv.printCoVisit(new File(coOutFile));
		} catch (Exception e1) {
			e1.printStackTrace();
		}

		// 与Embedding结果进行对比
		List<String> students = new ArrayList<>(s);
		for (File file : fileList) {
			Map<String, Double[]> emb = u.readEmb(file);
			int k = 2;
			MRR mrr = new MRR(students, mrrInterval, k, coVisit, emb);
			try {
				Map<Integer, Double> result = mrr.calculate(new File(resultFile), file.getName());
				mrr.setK(4);
				result = mrr.calculate(new File(resultFile), file.getName());
				mrr.setK(6);
				result = mrr.calculate(new File(resultFile), file.getName());
				mrr.setK(8);
				result = mrr.calculate(new File(resultFile), file.getName());
				mrr.setK(10);
				result = mrr.calculate(new File(resultFile), file.getName());
				mrr.setK(12);
				result = mrr.calculate(new File(resultFile), file.getName());
			} catch (Exception e1) {
				e1.printStackTrace();
			}
		}
		long mEnd = System.currentTimeMillis();
		System.out.println("程序共运行" + (mEnd - tStart) / 1000 + "秒");

	}

}
