package coVisitation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import Utils.Utils;
import dataProcess.Model;

public class CoVisit_org {

	private List<Model> records;
	private Map<String, Integer> coVisit;
	private Set<String> students;

	public CoVisit_org(File file, Date begin, Date end) {
		Utils u = new Utils();
		// records = u.readRecords(true, file, begin, end); // 输入文件必须按时间排列
		coVisit = new HashMap<>();
		students = new TreeSet<>();
	}

	// 带参数的getCoVisit方法
	public Map<String, Integer> getCoVisit(int seconds) {
		if (coVisit.isEmpty()) {
			coVisitNum(seconds);
			mergeCoVisit();
		}
		return coVisit;
	}

	// 按<"id1, id2", num>的形式返回两学生的共现次数
	public void coVisitNum(int seconds) {

		for (int i = 0; i < records.size(); i++) {
			students.add(records.get(i).getId());
			if (i % 10000 == 0)
				System.out.println(i / 10000);
			Set<String> s = new TreeSet<>();
			for (int j = i + 1; j < records.size(); j++) {
				if (records.get(i).getId().equals(records.get(j).getId())) {
					records.remove(j);
					continue;
				}
				if (records.get(j).getT().getTime() - records.get(i).getT().getTime() < seconds * 1000) {
					if (records.get(j).getLocation().equals(records.get(i).getLocation())) {
						String key = records.get(i).getId() + "," + records.get(j).getId();
						s.add(key);
					} else
						continue;
				} else
					break;
			}
			for (String str : s) {
				Integer count = coVisit.get(str);
				if (count == null)
					coVisit.put(str, 1);
				else
					coVisit.put(str, count + 1);
			}
		}
	}

	// 合并需要的数据，即对称的key值
	public void mergeCoVisit() {
		Iterator<HashMap.Entry<String, Integer>> it = coVisit.entrySet().iterator();
		for (; it.hasNext();) {
			HashMap.Entry<String, Integer> e = it.next();
			String key = e.getKey();
			String[] pair = key.split(",");
			String reverseKey = pair[1] + "," + pair[0];
			Integer num;
			if ((num = coVisit.get(reverseKey)) != null) {
				coVisit.put(reverseKey, e.getValue() + num);
				it.remove();
			}
		}
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
		for (Map.Entry<String, Integer> i : coVisit.entrySet()) {
			osw.write(i.getKey() + "," + i.getValue() + "\n");
		}
		osw.close();
		fos.close();
	}

	public List<Model> getRecords() {
		return records;
	}

	public void setRecords(List<Model> records) {
		this.records = records;
	}

	public Map<String, Integer> getCoVisit() {
		return coVisit;
	}

	public void setCoVisit(Map<String, Integer> coVisit) {
		this.coVisit = coVisit;
	}

	public Set<String> getStudents() {
		return students;
	}

	public void setStudents(Set<String> students) {
		this.students = students;
	}

}
