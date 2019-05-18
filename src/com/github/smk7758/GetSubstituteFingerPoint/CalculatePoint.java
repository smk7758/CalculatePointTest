package com.github.smk7758.GetSubstituteFingerPoint;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import com.github.smk7758.GetSubstituteFingerPoint.Main.LogLevel;

public class CalculatePoint {
	private Mat cameraMatrix;
	private MatOfDouble distortionCoefficients;
	double focus;

	public CalculatePoint(CameraParameter cameraParameter) {
		cameraMatrix = cameraParameter.cameraMatrix;
		distortionCoefficients = cameraParameter.distortionCoefficients_;

		double[] tmp_f_x = new double[cameraMatrix.channels()], tmp_f_y = new double[cameraMatrix.channels()];
		cameraMatrix.get(0, 0, tmp_f_x);
		cameraMatrix.get(1, 1, tmp_f_y);
		double f_x = tmp_f_x[0], f_y = tmp_f_y[0];
		focus = (f_x + f_y) / 2;
	}

	/**
	 * 指先の点の空間座標を変換して、接地判定を行う。
	 *
	 * @param vectorA_ 指先の点(Z=f)
	 */
	public void process(Point fingerPoint, Mat rotationVector, Mat translationVector, Mat outputMat) {
		Main.debugLog("Rv: " + rotationVector.dump(), LogLevel.DEBUG, "process | CalculatePoint");

		Mat rotationMatrix = new Mat();
		Calib3d.Rodrigues(rotationVector, rotationMatrix);
		Main.debugLog("R: " + rotationMatrix.dump(), LogLevel.DEBUG, "process | CalculatePoint");

		Mat translationVector_ = convertTranslationVector(translationVector);

		// A_, 指先の点(Z=f)
		Mat vectorA_ = new Mat(3, 1, CvType.CV_64F);
		vectorA_.put(0, 0, new double[] { fingerPoint.x, fingerPoint.y, focus });
		Main.debugLog("vectorA_: " + vectorA_.dump(), LogLevel.DEBUG, "process | CalculatePoint");

		// create Nw
		Mat vectorNw = new Mat(3, 1, CvType.CV_64F);
		vectorNw.put(0, 0, new double[] { 0, 0, 1 });

		// Nw → N
		Mat vectorN = new Mat();
		CalculatePoint.convertToWorldCoordinate(vectorNw, rotationMatrix, translationVector_, vectorN);

		double k_0 = vectorNw.dot(translationVector);
		double k_1 = vectorNw.dot(vectorA_);
		double k = vectorNw.dot(translationVector) / vectorNw.dot(vectorA_);

		Main.debugLog("k_0:  " + k_0 + ", k_1: " + k_1 + ", " + k, LogLevel.DEBUG, "process | CalculatePoint"); // TODO

		Mat vectorA = new Mat(3, 1, CvType.CV_64F);
		vectorA = vectorA_.mul(Mat.ones(vectorA_.size(), vectorA_.type()), k); // TODO

		Main.debugLog("A:  " + vectorA.dump(), LogLevel.DEBUG, "process | CalculatePoint");
		Main.debugLog("A_: " + vectorA_.dump(), LogLevel.DEBUG, "process | CalculatePoint");

		Mat vectorAw = new Mat(3, 1, CvType.CV_64F);

		// System.out.println(CvType.typeToString(rotationMatrix.type()));

		CalculatePoint.unconvertToWorldCoordinate(vectorA, rotationVector, translationVector_, vectorAw);

		Main.debugLog("Aw: " + vectorAw.dump(), LogLevel.DEBUG, "process | CalculatePoint"); // TODO

		int l = 20;
		Mat vectorBw = vectorAw.clone();
		Main.debugLog("Bw: " + vectorBw.dump(), LogLevel.DEBUG, "process | CalculatePoint");
		Main.debugLog("item: " + vectorBw.get(2, 0)[0], LogLevel.DEBUG, "process | CalculatePoint");

		vectorBw.put(0, 2, vectorBw.get(2, 0)[0] + l);

		Point3 pointA = new Point3(vectorA.get(0, 0)[0], vectorA.get(1, 0)[0], vectorA.get(2, 0)[0]);
		MatOfPoint3f pointsSrc = new MatOfPoint3f(pointA);
		MatOfPoint2f pointsDst = new MatOfPoint2f();
		Calib3d.projectPoints(pointsSrc, rotationVector, translationVector,
				cameraMatrix, distortionCoefficients, pointsDst);

		Main.debugLog("pointDst: " + pointsDst.get(0, 0)[0] + ", " + pointsDst.get(0, 0)[1],
				LogLevel.DEBUG, "process - CalculatePoint");

		Imgproc.circle(outputMat, new Point(pointsDst.get(0, 0)), 10, new Scalar(255, 255, 255));
		// System.out.println(pointsDst.dump()); // TODD
	}

	public static void convertToWorldCoordinate(Mat matSrc, Mat rotationMatrix, Mat translationVector, Mat matDst) {
		Main.debugLog("matSrc: " + matSrc.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");
		Main.debugLog("t: " + translationVector.t().dump(), LogLevel.DEBUG, "convertToWorldCoordinate");

		Core.gemm(rotationMatrix, matSrc, 1, translationVector.t(), 1, matDst);

		// ImgProcessUtil.multiplicationMat(rotationMatrix, matSrc, matDst);
		// Core.multiply(rotationVoctor, matSrc, matDst);
		// Core.add(matDst, translationVector, matDst);
	}

	public static void unconvertToWorldCoordinate(Mat matSrc, Mat rotationVector, Mat translationVector, Mat matDst) {
		Main.debugLog("matSrc: " + matSrc.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");
		Main.debugLog("R: " + rotationVector.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");
		Main.debugLog("t: " + translationVector.t().dump(), LogLevel.DEBUG, "convertToWorldCoordinate");

		// Mat rotationMatrix = new Mat();

		// Core.invert(rotationMatrix, rotationMatrix_0);
		// Imgproc.invertAffineTransform(rotationMatrix_0, rotationMatrix_0); // ゴミ

		Mat rotationVector_0 = rotationVector.mul(Mat.eye(rotationVector.size(), rotationVector.type()), -1);

		Mat rotationMatrix_0 = new Mat();
		Calib3d.Rodrigues(rotationVector_0, rotationMatrix_0);
		Main.debugLog("R0: " + rotationMatrix_0.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");

		Core.subtract(matSrc, translationVector.t(), matDst);

		// System.out.println("Rv0: " + rotationVector_0.dump() + ", " + rotationVector_0.height());

		Main.debugLog("matDst(before, multi): " + matDst.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");

		Core.gemm(rotationMatrix_0, matDst, 1, new Mat(), 0, matDst);
		// ImgProcessUtil.multiplicationMat(rotationVector_0, matDst, matDst);
		// Core.multiply(rotationMatrix_0, matDst, matDst); // TODO

		Main.debugLog("matDst(after, multi): " + matDst.dump(), LogLevel.DEBUG, "convertToWorldCoordinate");
	}

	/**
	 * Mat型のtranslationVectorの横ベクトルを返す。
	 *
	 * @param translationVector
	 * @return
	 */
	private static Mat convertTranslationVector(Mat translationVector) {
		Mat translationVector_ = new Mat(1, 3, CvType.CV_64F);
		for (int channel = 0; channel < translationVector.channels(); channel++) {
			translationVector_.put(0, channel, translationVector.get(0, 0)[channel]);
		}
		return translationVector_;
	}
}
