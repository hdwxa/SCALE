package dataProcess;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import Utils.Utils;

public class Process {

	private Set<String> students;
	private List<Model> records;
	private Map<Model, String> eventMap; // 用于查找对应事件编号
	private Map<String, String> locationMap; // 用于地点的id映射
	private Date begin;
	private Date end;
	private int stepHours;
	private Utils util;

	public List<Model> getRecords() {
		return records;
	}

	public void setRecords(List<Model> records) {
		this.records = records;
	}

	public Map<Model, String> getEventMap() {
		return eventMap;
	}

	public void setEventMap(Map<Model, String> eventMap) {
		this.eventMap = eventMap;
	}

	public Map<String, String> getLocationMap() {
		return locationMap;
	}

	public void setLocationMap(Map<String, String> locationMap) {
		this.locationMap = locationMap;
	}

	public Date getBegin() {
		return begin;
	}

	public void setBegin(Date begin) {
		this.begin = begin;
	}

	public Date getEnd() {
		return end;
	}

	public void setEnd(Date end) {
		this.end = end;
	}

	public int getStepHours() {
		return stepHours;
	}

	public void setStepHours(int stepHours) {
		this.stepHours = stepHours;
	}

	public Utils getUtil() {
		return util;
	}

	public void setUtil(Utils util) {
		this.util = util;
	}

	public void init(File file, String begin, String end, int stepHours, int stuSize) {
		util = new Utils();
		Date b = util.converTime(begin);
		Date e = util.converTime(end);
		records = new ArrayList<>();
		records = util.readRecords(records, true, file, b, e);
		Iterator<Model> it = records.iterator();
		eventMap = new HashMap<>();
		locationMap = new HashMap<>();
		locationMap.put("number", "0");
		this.begin = b;
		this.end = e;
		this.stepHours = stepHours;
	}

	@SuppressWarnings("deprecation")
	public String getTimeId(Date time) {
		int week = time.getDay() + 1;
		int slot = time.getHours() / stepHours;
		// if (slot == -1)
		// slot = 24 / stepHours - 1;
		return "T_" + week + "_" + slot;
	}

	// 若当前地点未存在locationMap中，则更新
	public String updateLocationMap(String location) {
		int number = Integer.parseInt(locationMap.get("number")) + 1;
		locationMap.put(location, "L_" + number);
		locationMap.put("number", number + "");
		return "L_" + number;
	}

	public String updateEventMap(Model record) {
		String tId = getTimeId(record.getT());
		String lId = locationMap.get(record.getLocation());
		if (lId == null) {
			lId = updateLocationMap(record.getLocation());
		}
		eventMap.put(record, tId + lId);
		return tId + lId;
	}

	public String getEventMap(Model record) {
		String eId = eventMap.get(record);
		if (eId == null)
			eId = updateEventMap(record);
		return eId;
	}

	// 处理为Simpsim输入, 输入文件需要按学生id排序
	public Map<String, List<String>> process(File file) {
		Map<String, List<String>> consumMap = new HashMap<>();
		List<String> events = new ArrayList<>();
		String id = records.get(0).getId();
		int count = 0;
		for (Model i : records) {
			if (id.equals(i.getId())) {
				// 添加事件
				events.add(getEventMap(i));
			} else {
				// 将上一个学生的事件添加进consumMap
				// consumMap.put(id, iEvents);
				printList(file, id, events);
				events = new ArrayList<>();
				id = i.getId();
				count++;
				System.out.println("第" + count + "个学生开始处理");
				events.add(getEventMap(i));
			}
		}
		printList(file, id, events);
		return consumMap;
	}

	// 将Map打印到文件
	public void printMap(File file, Map<String, String> map) {
		FileOutputStream fos = null;
		try {
			if (!file.exists()) {
				file.createNewFile();// 如果文件不存在，就创建该文件
				fos = new FileOutputStream(file);// 首次写入获取
			} else {
				// 如果文件已存在，那么就在文件末尾追加写入
				fos = new FileOutputStream(file, false);// 这里构造方法多了一个参数true,表示在文件末尾追加写入
			}
			OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");// 指定以UTF-8格式写入文件
			for (Map.Entry<String, String> e : map.entrySet()) {
				osw.write(e.getKey() + "\t" + e.getValue() + "\n");
			}
			osw.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// 将Map打印到文件
	public void printMapList(File file, Map<String, List<String>> map) {
		FileOutputStream fos = null;
		try {
			if (!file.exists()) {
				file.createNewFile();// 如果文件不存在，就创建该文件
				fos = new FileOutputStream(file);// 首次写入获取
			} else {
				// 如果文件已存在，那么就在文件末尾追加写入
				fos = new FileOutputStream(file, true);// 这里构造方法多了一个参数true,表示在文件末尾追加写入
			}
			OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");// 指定以UTF-8格式写入文件
			for (Map.Entry<String, List<String>> e : map.entrySet()) {
				for (String i : e.getValue()) {
					osw.write(e.getKey() + "\t" + i + "\n");
				}
			}
			osw.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	// 将List打印到文件
	public void printList(File file, String id, List<String> events) {
		FileOutputStream fos = null;
		try {
			if (!file.exists()) {
				file.createNewFile();// 如果文件不存在，就创建该文件
				fos = new FileOutputStream(file);// 首次写入获取
			} else {
				// 如果文件已存在，那么就在文件末尾追加写入
				fos = new FileOutputStream(file, true);// 这里构造方法多了一个参数true,表示在文件末尾追加写入
			}
			OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");// 指定以UTF-8格式写入文件
			for (String e : events) {
				osw.write(id + "\t" + e + "\n");
			}
			osw.close();
			fos.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
