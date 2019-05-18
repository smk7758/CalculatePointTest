package com.github.smk7758.GetSubstituteFingerPoint;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

public class ImgProcessUtil {
	private ImgProcessUtil() {
	}

	public static void multiplicationMat(Mat matSrc0, Mat matSrc1, Mat matDst) {
		Core.gemm(matSrc0, matSrc1, 1, new Mat(), 0, matDst);
	}

	public static double[] getCenter(List<Point> points) {
		final MatOfPoint points_ = new MatOfPoint();
		points_.fromList(points);
		return getCenter(points_);
	}

	public static double[] getCenter(MatOfPoint points) {
		// 重心を取得(投げてるver)
		Moments moments = Imgproc.moments(points);
		double[] center = { moments.get_m10() / moments.get_m00(), moments.get_m01() / moments.get_m00() };
		return center;
	}

	public static int getLargestArea(List<MatOfPoint> contours) {
		double area_max = 0, area = 0;
		int contour_max_area = -1;
		for (int j = 0; j < contours.size(); j++) {
			// area = contours.get(j).size().area(); //コッチはその面積のため違う
			area = Imgproc.contourArea(contours.get(j)); // 境界点の面積
			if (area_max < area) {
				area_max = area;
				contour_max_area = j;
			}
		}
		if (contour_max_area < 0) System.err.println("Cannot find contours area.");
		return contour_max_area;
	}

	/**
	 * 大きい順にソートして、面積も共に返す。
	 *
	 * @param contours 境界線の集合
	 * @return 大きい順にソートして、contoursでのIndexと、その境界線の面積。
	 */
	public static ListMap<Integer, Double> getSortedAreaNumber(List<MatOfPoint> contours) {
		// if (contours.size() < 1) throw new IllegalArgumentException("No countrus.");
		double area = 0;
		ListMap<Integer, Double> contour_area = new ListMap<>();
		for (int i = 0; i < contours.size(); i++) {
			// area = contours.get(j).size().area(); //コッチはその面積のため違う ??
			area = Imgproc.contourArea(contours.get(i)); // 普通の面積
			if (contour_area.size() < 1 || contour_area.getValue(0) < area) {
				contour_area.add(0, i, area);
			} else {
				contour_area.add(i, area);
			}
		}
		return contour_area;
	}

	/**
	 * @param mat 処理を行う元となる画像。
	 * @param contours 境界線の集合
	 * @return 領域の中心。
	 */
	public static Optional<List<double[]>> getCenterPointContrus(Mat mat, List<MatOfPoint> contours) {
		List<double[]> result = new ArrayList<>();
		// List<MatOfPoint> contours = new ArrayList<>(); // 境界線の集合

		// 境界線を点としてとる(輪郭線)
		Imgproc.findContours(mat, contours, new Mat(mat.size(), mat.type()), Imgproc.RETR_EXTERNAL,
				Imgproc.CHAIN_APPROX_NONE);

		if (contours.size() < 1) {
			System.out.println("Cannot find any point! : after findContours @getCenterPointContrus");
			return Optional.empty();
		}

		ListMap<Integer, Double> contours_area = ImgProcessUtil.getSortedAreaNumber(contours);

		if (contours_area.size() < 0) {
			System.err.println("Cannot find any point! : after getLargestAreaNumber @getCenterPointContrus");
			return Optional.empty();
		}

		for (int i = 0; i < contours_area.size(); i++) {
			MatOfPoint contour_points = contours.get(contours_area.get(i).getKey());
			// 重心
			result.add(ImgProcessUtil.getCenter(contour_points));
		}
		return Optional.ofNullable(result);
	}

	public static Optional<List<double[]>> getCenterPointContrus(Mat mat) {
		List<MatOfPoint> contours = new ArrayList<>(); // 境界線の集合
		return getCenterPointContrus(mat, contours);
	}

	public static List<MatOfPoint> getContrus(Mat mat) {
		List<MatOfPoint> result = new ArrayList<>();
		List<MatOfPoint> contours = new ArrayList<>();

		// 境界線を点としてとる(輪郭線)
		Imgproc.findContours(mat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

		if (contours.size() < 1) {
			System.out.println("Cannot find any point! : after findContours");
			return null;
		}

		ListMap<Integer, Double> contours_area = ImgProcessUtil.getSortedAreaNumber(contours);

		if (contours_area.size() < 0) {
			System.err.println("Cannot find any point! : after getLargestAreaNumber");
			return null;
		}

		for (int i = 0; i < contours_area.size(); i++) {
			MatOfPoint contour_points = contours.get(contours_area.get(i).getKey());
			result.add(contour_points);
		}
		// 重心
		return result;
	}

	public static int getSmallestInclinationNumber(final MatOfPoint points, int[] hull_array) {
		List<Double> inclination_hull_points = new ArrayList<>(); // このインデックスはhullの点のインデックスと対応する。(ex: 1 -> (1, 2))

		for (int i = 0; i < hull_array.length - 1; i++) {
			final double inclination = getInclination(points.get(hull_array[i], 0), points.get(hull_array[i + 1], 0));
			// System.out.println("Inclination: " + inclination);
			inclination_hull_points.add(inclination);
		}

		// 最小傾きの取得
		int smallest = 0;
		for (int i = 0; i < inclination_hull_points.size(); i++) {
			if (Math.abs(inclination_hull_points.get(smallest)) > Math.abs(inclination_hull_points.get(i))) {
				smallest = i;
			}
		}
		// System.out.println("Smallest: " + smallest + ", size: " + inclination_hull_points.size());
		return smallest;
	}

	public static double getInclination(double[] first_point, double[] second_point) {
		return (second_point[1] - first_point[1]) / (second_point[0] - first_point[0]);
	}
}
