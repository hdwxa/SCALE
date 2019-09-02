package Utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import dataProcess.Model;

public class Utils {

	// 读原始记录的csv文件
	public List<Model> readRecords(List<Model> dataList, boolean header, File file, Date begin, Date end) {
		BufferedReader br = null;
		try {
			br = new BufferedReader(new FileReader(file));
			String line = "";
			if (header == true)
				line = br.readLine();
			while ((line = br.readLine()) != null) {
				Model record = new Model();
				String[] l = line.split(",");
				Date t = converTime(l[1]);
				if (t.before(begin) || t.after(end))
					continue;
				record.setId(l[0]);
				record.setT(t);
				record.setMoney(Double.parseDouble(l[2]));
				record.setLocation(l[3]);
				record.setType(l[4]);
				record.setRemark(l[5]);
				dataList.add(record);
			}
		} catch (Exception e) {
		} finally {
			if (br != null) {
				try {
					br.close();
					br = null;
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return dataList;
	}

	// 读embedding结果
	public Map<String, Double[]> readEmb(File file) {
		Map<String, Double[]> emb = new HashMap<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line = br.readLine();
			String[] head = line.split(" ");
			Double[] info = new Double[2];
			info[0] = (double) Integer.parseInt(head[0]);
			info[1] = (double) Integer.parseInt(head[1]);
			int dim = (int) info[1].doubleValue();
			emb.put("info", info);
			while ((line = br.readLine()) != null) {
				String[] l = line.split(" ");
				String id = l[0];
				Double[] vec = new Double[dim];
				for (int i = 0; i < dim; i++) {
					vec[i] = Double.parseDouble(l[i + 1]);
				}
				emb.put(id, vec);
			}
			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return emb;
	}

	public List<String> readStudent(File file) {
		List<String> students = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while ((line = br.readLine()) != null) {
				String[] l = line.split(",");
				students.add(l[0]);
			}

			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return students;
	}

	public Map<String, List<String>> readTopk(File file, int k) {
		Map<String, List<String>> students = new HashMap<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while ((line = br.readLine()) != null) {
				String[] l = line.split(",");
				List<String> kStudents = new ArrayList<>();
				for (int i = 1; i <= k; i++) {
					kStudents.add(l[i]);
				}
				students.put(l[0], kStudents);
			}

			br.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return students;
	}

	// string to Date
	public Date converTime(String str) {
		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
		try {
			return formatter.parse(str);
		} catch (ParseException e) {
			e.printStackTrace();
		}
		return null;
	}

	// 获得日期在一天中的时刻
	@SuppressWarnings("deprecation")
	public static int getSeconds(Date date) {
		return date.getHours() * 3600 + date.getMinutes() * 60 + date.getSeconds();
	}

	// 计算两个时间相差多少个小时
	public static long diffHour(Date date1, Date date2) {
		// long nd = 1000 * 24 * 60 * 60;
		long nh = 1000 * 60 * 60;
		long diff = date1.getTime() - date2.getTime();
		return diff / nh;
	}
}
