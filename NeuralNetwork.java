///////////////////////////////// FILE HEADER /////////////////////////////////
//
// Title:           UWMadison_CS540_Su20_P01
// This File:       NeuralNetwork.java
// Files:           LogisticRegression.java, NeuralNetwork.java,
//                  Dataset.java, NeuralNetworkLayer.java,
//                  NeuralNetworkFunction.java
// External Class:  None
//
// GitHub Repo:     https://github.com/hyecheol123/UWMadison_CS540_Su20_P01
//
// Author
// Name:            Hyecheol (Jerry) Jang
// Email:           hyecheol.jang@wisc.edu
// Lecturer's Name: Young Wu
// Course:          CS540 (LEC 002 / Epic), Summer 2020
//
///////////////////////////// OUTSIDE REFERENCE  //////////////////////////////
//
// List of Outside Reference
//   1.
//
////////////////////////////////// KNOWN BUGS /////////////////////////////////
//
// List of Bugs
//   1.
//
/////////////////////////////// 80 COLUMNS WIDE //////////////////////////////

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import static java.lang.System.exit;

/**
 * Contain codes and relevant main method for Neural Network
 * having 1 hidden layer with 392 units (with logistic activation function)
 */
public class NeuralNetwork {
  // Class variable
  static Dataset dataset;
  // Variables for NeuralNetwork
  static ArrayList<NeuralNetworkLayer> layers;
  // For checking convergence
  static double currentLoss = 0;
  static double previousLoss = 0;
  // Counting training epochs
  static int epoch;

  // Constants
  final static String CSV_FILE_LOCATION = "mnist_train.csv";
  final static String TEST_SET_LOCATION = "test.txt";
  final static int LABEL0 = 4;
  final static int LABEL1 = 7;
  final static int NUM_HIDDEN_LAYER = 1;
  final static int NUM_FEATURE = 784; // MNIST
  final static int[] NUM_UNITS = {392, 1}; // last entry is the number of output
  final static long RANDOM_SEED = 2006031213;
  // Hyper-parameters
  final static double EPSILON = 0.000001;
  final static int MAX_EPOCH = 50000;
  final static double LEARNING_RATE = 0.1;

  /**
   * main method of NeuralNetwork Class
   *
   * @param args Command Line Arguments (CLAs)
   *             0. Question 5 save file name ()
   *             1. Question 6 save file name ()
   *             2. Question 7 save file name ()
   *             3. Question 8 save file name ()
   *             4. Question 9 save file name ()
   *             5. log file
   * @throws IOException while writing file, IOException might be occurred.
   */
  public static void main(String[] args) throws IOException {
    // Check for the number of command=line arguments
    if(args.length != 6) {
      System.out.println("Need to have Six CLAs, the file names to store text file for each question and for logging.");
      exit(1);
    }

    // Create log file
    File log = new File(args[5]);
    if(!log.exists()) { // only when file not exists
      if(!log.createNewFile()) { // creating new file with given name
        System.out.println("File not created!!");
        exit(1);
      }
    }
    FileWriter logWriter = new FileWriter(log);

    // Create dataset and load data
    dataset = new Dataset(CSV_FILE_LOCATION, TEST_SET_LOCATION, LABEL0, LABEL1, NUM_FEATURE);
    dataset.loadTrainingSet();
    dataset.loadTestingSet();

    // Initialize Layers with random weights and bias ([-1, 1])
    Random random = new Random(RANDOM_SEED);
    layers = new ArrayList<>(NUM_HIDDEN_LAYER + 1); // hidden layers + output layer
    layers.set(0, new NeuralNetworkLayer(NUM_FEATURE, NUM_UNITS[0], random, -1, 1));
    for(int layerIndex = 1; layerIndex < NUM_HIDDEN_LAYER + 1; layerIndex++) {
      layers.set(layerIndex, new NeuralNetworkLayer(NUM_UNITS[layerIndex - 1], NUM_UNITS[layerIndex], random, -1, 1));
    }

    // Calculate activations of each layer
    ArrayList<ArrayList<Double[]>> activations = new ArrayList<>(NUM_HIDDEN_LAYER + 1);
    activations.set(0, calculateActivation(dataset.getTrainingFeatures(),
        NUM_UNITS[0], layers.get(0))); // first layer (from feature to 1st hidden)
    for(int layerIndex = 1; layerIndex < NUM_HIDDEN_LAYER + 1; layerIndex++) { // for the other layers
      activations.set(layerIndex, calculateActivation(activations.get(layerIndex - 1),
          NUM_UNITS[layerIndex], layers.get(layerIndex)));
    }

    // Gradient Descent, Training Neural Network
    for(epoch = 1; epoch <= MAX_EPOCH; epoch++) {
      // logging
      System.out.print("Epoch " + epoch + " ");
      logWriter.write("Epoch" + epoch + " ");

      // Update Weights and Bias
      updateWeightsAndBias(activations);

      // Calculate New Activation
      activations.set(0, calculateActivation(dataset.getTrainingFeatures(),
          NUM_UNITS[0], layers.get(0))); // first layer (from feature to 1st hidden)
      for(int layerIndex = 1; layerIndex < NUM_HIDDEN_LAYER + 1; layerIndex++) { // for the other layers
        activations.set(layerIndex, calculateActivation(activations.get(layerIndex - 1),
            NUM_UNITS[layerIndex], layers.get(layerIndex)));
      }

      // Check for convergence
      if(checkConvergence(activations.get(NUM_HIDDEN_LAYER), logWriter)) { // when converge
        System.out.println("Converged after Epoch " + epoch);
        logWriter.write("Converged after Epoch " + epoch + "\n");
        break;
      }

      // Flush log occasionally
      if(epoch % 10 == 0) {
        logWriter.flush();
      }
    }

    // Flush log
    logWriter.flush();
    logWriter.close();

    // TODO Questions
  }

  /**
   * Private helper method to calculate activation
   *
   * @param prevActivation ArrayList of Double array containing activation from the previous layer
   *                       (if this is first hidden layer, get features).
   *                       Each entry of ArrayList contains activation calculated for each data point
   * @param numUnits number of units in current layer
   * @param layer NeuralNetworkLayer object that contains weights and bias information
   * @return ArrayList of Double array containing activation for current hidden layer
   */
  private static ArrayList<Double[]> calculateActivation(ArrayList<Double[]> prevActivation,
                                                         int numUnits, NeuralNetworkLayer layer) {
    int numDataEntries = prevActivation.size();
    ArrayList<Double[]> activation = new ArrayList<>(numDataEntries); // place to store activation
    // retrieve weights and bias
    Double[][] weights = layer.getWeights();
    Double[] bias = layer.getBias();

    for(int dataIndex = 0; dataIndex < numDataEntries; dataIndex++) { // for all data entry
      activation.set(dataIndex, new Double[numUnits]); // set the Double array storing activation

      // calculate activation for each hidden Units
      for(int hiddenUnitIndex = 0; hiddenUnitIndex < numUnits; hiddenUnitIndex++) {
        // calculate weighted sum
        double weightedSum = 0;
        for(int prevActivationIndex = 0; prevActivationIndex < prevActivation.get(dataIndex).length;
            prevActivationIndex++) {
          weightedSum +=
              weights[prevActivationIndex][hiddenUnitIndex] * prevActivation.get(dataIndex)[prevActivationIndex];
        }
        weightedSum += bias[hiddenUnitIndex];

        // calculate activation based on the weighted sum
        activation.get(dataIndex)[hiddenUnitIndex] = NeuralNetworkFunction.logisticSigmoid(weightedSum);
      }
    }

    return activation;
  }

  /**
   * Private helper method to update Weights and Bias of the neural network
   *
   * @param activations previously calculated activation
   *                    Outer ArrayList - Indicates each layer
   *                    Inner ArrayList - Indicates data points
   *                    Double[] Array - The activation of the unit
   */
  private static void updateWeightsAndBias(ArrayList<ArrayList<Double[]>> activations) {
    // TODO Implements
  }

  /**
   * Private helper method to check convergence by calculating loss function
   *
   * @param output    output with updated weight and bias
   * @param logWriter FileWriter of log file
   * @return whether the model has been converged or not
   * @throws IOException while writing log file, IOException might be occurred.
   */
  private static boolean checkConvergence(ArrayList<Double[]> output, FileWriter logWriter) throws IOException {
    previousLoss = currentLoss; // assign previous loss

    // calculate current loss
    currentLoss = 0;
    for(int dataIndex = 0; dataIndex < output.size(); dataIndex++) {
      currentLoss += NeuralNetworkFunction.binaryCrossEntropy(
          dataset.getTrainingLabels().get(dataIndex), output.get(dataIndex)[0]);
    }

    // logging
    System.out.println("Loss: " + currentLoss);
    logWriter.write("Loss: " + currentLoss + "\n");

    // Check for convergence
    return Math.abs(previousLoss - currentLoss) < EPSILON;
  }

}
