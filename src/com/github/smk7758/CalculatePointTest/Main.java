package com.github.smk7758.CalculatePointTest;

import java.nio.file.Paths;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;

public class Main {
	private static final boolean DEBUG_MODE = true;
	private final String camparaPathString = "S:\\FingerPencil\\CalclatePoint_Test_2019-05-18\\CameraCalibration_Test_2019-05-18.xml";

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	public static void main(String[] args) {
		new Main().processer();
	}

	public enum LogLevel {
		ERROR, WARN, INFO, DEBUG
	}

	public static void debugLog(String message, LogLevel logLevel, String fromSuffix) {
		debugLog(message + " @" + fromSuffix, logLevel);
	}

	public static void debugLog(String message, LogLevel logLevel) {
		if (DEBUG_MODE) System.out.println("[" + logLevel.toString() + "] " + message);
	}

	public void processer() {
		CameraParameter cameraParameter = new CameraParameter(Paths.get(camparaPathString));
		CalculatePoint calculatePoint = new CalculatePoint(cameraParameter);

		// Point fingerPoint = new Point(0, 0);

		Mat vectorA_ = new Mat(new Size(1, 3), CvType.CV_64FC1);
		vectorA_.put(0, 0, new double[] { 0, 2, 1 });

		Mat vectorB_ = new Mat(new Size(1, 3), CvType.CV_64FC1);
		vectorB_.put(0, 0, new double[] { 0, 1, 1 });

		Mat rotationMatrix = Mat.eye(new Size(3, 3), CvType.CV_64FC1);
		rotationMatrix.put(1, 1, new double[] { -1 });

		debugLog("R: " + rotationMatrix.dump(), LogLevel.DEBUG);

		Mat translationVector = new Mat(new Size(1, 3), CvType.CV_64FC1);
		translationVector.put(0, 0, new double[] { 0, 10, 1 });
		debugLog("t: " + translationVector.dump(), LogLevel.DEBUG);

		Mat outputMat = new Mat();

		calculatePoint.process_(vectorA_, vectorB_, rotationMatrix, translationVector, outputMat);
	}

}
