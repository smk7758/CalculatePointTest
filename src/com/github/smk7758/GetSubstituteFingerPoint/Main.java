package com.github.smk7758.GetSubstituteFingerPoint;

import java.nio.file.Paths;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;

public class Main {
	private static final boolean DEBUG_MODE = true;
	private final String cpPathString = "S:\\FingerPencil\\CalclatePoint_Test_2019-05-18\\CameraCalibration_Test_2019-05-18.xml";

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
		CameraParameter cameraParameter = new CameraParameter(Paths.get(cpPathString));
		CalculatePoint calculatePoint = new CalculatePoint(cameraParameter);

		Point fingerPoint = new Point(1, 2);
		Mat rotationMatrix = Mat.eye(new Size(3, 3), CvType.CV_64FC1);
		debugLog(rotationMatrix.dump(), LogLevel.DEBUG);
		Mat rotationVector = new Mat();
		{
			Calib3d.Rodrigues(rotationMatrix, rotationVector);
		}
		Mat translationVector = new Mat(new Size(3, 1), CvType.CV_64FC1);
		{
			translationVector.put(0, 0, new double[] { 0 });
			translationVector.put(0, 1, new double[] { 0 });
			translationVector.put(0, 2, new double[] { 10 });
		}
		debugLog(translationVector.dump(), LogLevel.DEBUG);
		Mat outputMat = new Mat();
		calculatePoint.process(fingerPoint, rotationVector, translationVector, outputMat);

	}

}
