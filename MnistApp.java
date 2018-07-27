package com.example;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class MnistApp {
	private static PrintStream out = System.out;

	private static int argmax(FloatBuffer buffer) {
		float[] values = buffer.array();
		int dim = values.length;
		int argmax = -1;
		float maxval = Float.MIN_VALUE;
		for (int j = 0; j < dim; ++j) {
			if (values[j] > maxval) {
				maxval = values[j];
				argmax = j;
			}
		}
		return argmax;
	}

	private static float accuracy(int[] y_true, int[] y_pred) {
		int m = y_true.length;
		int score = 0;
		for (int i = 0; i < m; ++i) {
			if (y_pred[i] == y_true[i]) {
				++score;
			}
		}
		return score * 1f / m;
	}

	private static String[] readFile(String path) {
		ArrayList<String> lines = new ArrayList<>();
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(path));
			String line;
			while ((line = reader.readLine()) != null) {
				lines.add(line);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return lines.toArray(new String[lines.size()]);
	}

	public static void main(String[] args) {

		String path = args[0];
		String modelPath = "/tmp/tf/model/saved_model.pb"; //args[1];

		final int dim = 784;
		final String[] lines = readFile(path);

		final int m = lines.length;
		final int[] y = new int[m];
		final int[][] X = new int[m][dim];

		for (int i = 0; i < m; ++i) {
			String[] tokens = lines[i].split(",");
			y[i] = Integer.parseInt(tokens[0]);
			for (int j = 0; j < dim; ++j) {
				X[i][j] = Integer.parseInt(tokens[j]);
			}
		}

		SavedModelBundle savedModelBundle = SavedModelBundle.load(modelPath, "serve");

		Session sess = savedModelBundle.session();

		final int[] y_pred = new int[m];
		final FloatBuffer outputBuffer = FloatBuffer.allocate(dim);
		Tensor<Integer> inputTensor;

		for (int i = 0; i < m; ++i) {
			inputTensor = Tensor.create(new long[] { 1, dim }, IntBuffer.wrap(X[i]));
			outputBuffer.rewind();
			sess.runner().feed("X:0", inputTensor).fetch("Y_prob:0").run().get(0).writeTo(outputBuffer);
			y_pred[i] = argmax(outputBuffer);
		}
		float accuracyScore = accuracy(y, y_pred);

		out.printf("Accuracy: %f", accuracyScore);
	}

}
