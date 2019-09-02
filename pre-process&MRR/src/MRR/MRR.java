package MRR;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

public class MRR {

	private List<String> queryStudents;
	private int mrrInterval;
	private int k; // 对比前k个最相似的学生，计算MRR
	private Map<String, Map<String, Integer>> coVisitNum; // 通过coVisit类students获取
	private Map<String, Double[]> emb; // 通过Utils类的readEmb方式传入
	private int dim; // emb的维数
	private Map<String, Map<String, Integer>> coVisitRank;
	private Map<String, List<String>> scale;
	private Map<String, Double> sMrrScore;

	public MRR(List<String> qs, int mi, int k, Map<String, Map<String, Integer>> c, Map<String, Double[]> e) {
		queryStudents = qs;
		mrrInterval = mi; // 每处理mi个学生，产生一次输出
		this.k = k;
		coVisitNum = c;
		emb = e;
		dim = (int) e.get("info")[1].doubleValue();
		coVisitRank = new HashMap<>();
		scale = new HashMap<>();
	}

	public void setK(int k) {
		this.k = k;
	}

	public Map<Integer, Double> calculate(File file, String key) throws Exception {
		calCoVisit();
		calScale();
		Map<Integer, Double> result = calMrr(file, key);
		return result;
	}

	// 根据CoVisit计算所有学生前k个相似学生， 可并行
	public void calCoVisit() {
		for (String id : queryStudents) {
			List<Map.Entry<String, Integer>> list = new ArrayList<>(coVisitNum.get(id).entrySet());

			Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
				// 降序排序
				public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2) {
					return -1 * o1.getValue().compareTo(o2.getValue());
				}

			});
			Map<String, Integer> topk = new HashMap<>();
			topk.put(id, 1);
			for (int i = 0; i < k && i < list.size(); i++) {
				topk.put(list.get(i).getKey(), i + 2);
			}
			coVisitRank.put(id, topk);
		}
	}

	public double calSimScore(Double[] s1, Double[] s2) {
		double mutiple = 0;
		double norm_vec1 = 0;
		double norm_vec2 = 0;
		for (int i = 0; i < dim; i++) {
			mutiple += s1[i] * s2[i];
			norm_vec1 += s1[i] * s1[i];
			norm_vec2 += s2[i] * s2[i];
		}
		if (norm_vec1 == 0 || norm_vec2 == 0)
			return 0;
		return mutiple / Math.sqrt(norm_vec1 * norm_vec2);
	}

	// 根据scale计算前k个相似学生, 可并行
	public void calScale() {
		Map<String, Map<String, Double>> simScore = new TreeMap<>();
		// System.out.println(queryStudents.size() + "2wqefweqf");
		for (String s1 : queryStudents) {
			Map<String, Double> s1SimScore = new TreeMap<>();
			for (String s2 : queryStudents) {
				if (simScore.get(s2) != null && simScore.get(s2).get(s1) != null) {
					s1SimScore.put(s2, simScore.get(s2).get(s1));
					continue;
				}
				s1SimScore.put(s2, calSimScore(emb.get(s1), emb.get(s2)));
			}
			simScore.put(s1, s1SimScore);
		}
		for (String id : queryStudents) {
			List<Map.Entry<String, Double>> list = new ArrayList<>(simScore.get(id).entrySet());

			Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
				// 降序排序
				public int compare(Entry<String, Double> o1, Entry<String, Double> o2) {
					return -1 * o1.getValue().compareTo(o2.getValue());
				}

			});
			List<String> topk = new ArrayList<>();
			for (int i = 0; i < k && i < list.size(); i++) {
				topk.add(list.get(i).getKey());
			}
			scale.put(id, topk);
		}

	}

	// 根据间隔mrrInterval计算MRR结果
	public Map<Integer, Double> calMrr(File file, String key) throws Exception {
		sMrrScore = new HashMap<>();
		Map<Integer, Double> mrrScore = new HashMap<>();
		Double sum = 0.0;
		for (String id : queryStudents) {
			// if (sMrrScore.size() % mrrInterval == 0 || sMrrScore.size() ==
			// queryStudents.size()) {
			// if(sMrrScore.size() != 0) {
			// mrrScore.put(sMrrScore.size(), sum);
			// System.out.println(sMrrScore.size() + sum + " ");
			// }
			// }
			Map<String, Integer> cv = coVisitRank.get(id);
			List<String> sc = scale.get(id);
			double score = 0, j = 1;
			for (int i = 0; i < k && i < cv.size() && i < sc.size(); i++, j *= 0.5) {
				Integer rank;
				if ((rank = cv.get(sc.get(i))) != null)
					score += (double) 1 / (i + 1);
			}
			sMrrScore.put(id, score);
			sum += score;
		}
		mrrScore.put(sMrrScore.size(), sum);
		saveMrrScore(file, key, mrrScore);
		return mrrScore;
	}

	public void saveMrrScore(File file, String key, Map<Integer, Double> mrrScore) throws Exception {
		FileOutputStream fos = null;
		if (!file.exists()) {
			file.createNewFile();// 如果文件不存在，就创建该文件
			fos = new FileOutputStream(file);// 首次写入获取
		} else {
			// 如果文件已存在，那么就在文件末尾追加写入
			fos = new FileOutputStream(file, true);// 这里构造方法多了一个参数true,表示在文件末尾追加写入
		}
		OutputStreamWriter osw = new OutputStreamWriter(fos, "UTF-8");// 指定以UTF-8格式写入文件

		// osw.write("MRR Score with interval " + mrrInterval + "\n");
		List<Map.Entry<Integer, Double>> list = new ArrayList<>(mrrScore.entrySet());

		Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
			// 升序排序
			public int compare(Entry<Integer, Double> o1, Entry<Integer, Double> o2) {
				return o1.getKey().compareTo(o2.getKey());
			}

		});
		osw.write(key + "\n");
		for (Map.Entry<Integer, Double> e : list) {
			osw.write(e.getValue() / e.getKey() + "");
		}
		osw.write("\n");
		// for (Map.Entry<String, Double> e : sMrrScore.entrySet()) {
		// osw.write(e.getKey() + " " + e.getValue() + "\n");
		// }
		osw.close();
		fos.close();
	}
}
