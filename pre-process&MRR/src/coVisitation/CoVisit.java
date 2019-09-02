package coVisitation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import Utils.Utils;
import dataProcess.Model;

public class CoVisit {

	private List<Model> records;
	private Map<String, Map<String, Integer>> coVisit;
	private Set<String> students;

	public CoVisit(File file1, File file2, Date begin, Date end) {
		Utils u = new Utils();
		records = new ArrayList<>();
		records = u.readRecords(records, true, file1, begin, end); // 输入文件必须按时间排列
		records = u.readRecords(records, true, file2, begin, end); // 输入文件必须按时间排列
		coVisit = new TreeMap<>();
		students = new TreeSet<>();
	}

	// 带参数的getCoVisit方法
	public Map<String, Map<String, Integer>> getCoVisit(int seconds) {
		if (coVisit.isEmpty()) {
			coVisitNum(seconds);
			mergeCoVisit();
		}
		return coVisit;
	}

	// 按<"id1, id2", num>的形式返回两学生的共现次数
	public void coVisitNum(int seconds) {
		System.out.println("总记录数为" + records.size());
		for (int i = 0; i < records.size(); i++) {
			students.add(records.get(i).getId());
			if (i % 10000 == 0)
				System.out.println(i / 10000);
			Set<String> s = new TreeSet<>();
			for (int j = i + 1; j < records.size(); j++) {
				if (Math.abs(records.get(j).getT().getTime() - records.get(i).getT().getTime()) < seconds * 1000) {
					if (records.get(j).getLocation().equals(records.get(i).getLocation())) {
						if (records.get(i).getId().equals(records.get(j).getId())) {
							records.remove(j);
							j--;
							continue;
						}
						String key = records.get(j).getId();
						s.add(key);
					} else
						continue;
				} else {
					break;
				}

			}
			Map<String, Integer> coVisitI = coVisit.get(records.get(i).getId());
			if (coVisitI == null)
				coVisitI = new TreeMap<>();
			for (String str : s) {

				Integer count = coVisitI.get(str);
				if (count == null)
					coVisitI.put(str, 1);
				else
					coVisitI.put(str, count + 1);
			}
			coVisit.put(records.get(i).getId(), coVisitI);
		}
		System.out.println("coVisit的大小为" + coVisit.size());
	}

	// 合并需要的数据，即对称的key值
	public void mergeCoVisit() {
		Map<String, Map<String, Integer>> result = new HashMap<>();
		for (String id1 : students) {
			Map<String, Integer> id1Co = new HashMap<>();
			if (result.get(id1) != null)
				id1Co = result.get(id1);
			for (Map.Entry<String, Integer> e : coVisit.get(id1).entrySet()) {
				String id2 = e.getKey();
				Integer temp;
				if ((temp = coVisit.get(id2).get(id1)) != null)
					id1Co.put(id2, e.getValue() + temp);
				else {
					id1Co.put(id2, e.getValue());
					if (result.get(id2) != null)
						result.get(id2).put(id1, e.getValue()); // 将result重置
					else {
						Map<String, Integer> id2Co = new HashMap<>();
						id2Co.put(id1, e.getValue());
						result.put(id2, id2Co);
					}
				}
			}
			result.put(id1, id1Co);
		}
		coVisit = result;
	}

	public void printCoVisit(File file) throws Exception {
		FileOutputStream fos = null;
		if (!file.exists()) {
			file.createNewFile();// 如果文件不存在，就创建该文件
			fos = new FileOutputStream(file);// 首次写入获取
		} else {
			// 如果文件已存在，那么就在文件末尾追加写入
			fos = new FileOutputStream(file, false);// 这里构造方法多了一个参数true,表示在文件末尾追加写入
		}
		OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");// 指定以UTF-8格式写入文件
		for (Map.Entry<String, Map<String, Integer>> i : coVisit.entrySet())
			for (Map.Entry<String, Integer> j : i.getValue().entrySet())
				osw.write(i.getKey() + "," + j.getKey() + "," + j.getValue() + "\n");
		osw.close();
		fos.close();
	}

	public List<Model> getRecords() {
		return records;
	}

	public void setRecords(List<Model> records) {
		this.records = records;
	}

	public Map<String, Map<String, Integer>> getCoVisit() {
		return coVisit;
	}

	public void setCoVisit(Map<String, Map<String, Integer>> coVisit) {
		this.coVisit = coVisit;
	}

	public Set<String> getStudents() {
		return students;
	}

	public void setStudents(Set<String> students) {
		this.students = students;
	}

}
