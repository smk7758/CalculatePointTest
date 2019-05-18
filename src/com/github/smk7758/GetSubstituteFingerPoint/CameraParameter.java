package com.github.smk7758.GetSubstituteFingerPoint;

import java.nio.file.Path;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

public class CameraParameter {
	public Mat cameraMatrix;
	public Mat distortionCoefficients;
	public MatOfDouble distortionCoefficients_;

	public CameraParameter(Path cameraParameterFilePath) {
		final Map<String, Mat> calibrationMats = MatIO.loadMat(cameraParameterFilePath);
		cameraMatrix = calibrationMats.get("CameraMatrix");
		distortionCoefficients = calibrationMats.get("DistortionCoefficients");
		distortionCoefficients_ = new MatOfDouble(distortionCoefficients);

		System.out.println("CameraMatrix: " + cameraMatrix.dump());
		System.out.println("DisCoeff_: " + distortionCoefficients_.dump());
	}
}
