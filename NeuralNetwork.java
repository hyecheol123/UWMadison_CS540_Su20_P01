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
// List of Bugs (and Possible Further Improvements)
//   1. updateWeightsAndBias() only works for one output case
//   2. checkConvergence() only work for one output case
//   3. Testing parts in main() only work for one output case
//   3. Rather than using hard coded parameters and hyper-parameter,
//          Need to find alternative ways to get parameter as CLAs in future.
//   4. Slow speed.
//          May boost up by using parallel computation (multi-thread),
//          or using third-party library to boost matrix multiplication
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
  // Hyper-parameters
  final static double EPSILON = 0.00000001;
  // For checking convergence
  static double currentLoss = 0;
  static double previousLoss = 0;
  final static int MAX_EPOCH = 5000;
  // Counting training epochs
  static int epoch;
  final static double LEARNING_RATE = 0.00001; // 0.0001

  // Constants
  final static String CSV_FILE_LOCATION = "mnist_train.csv";
  final static String TEST_SET_LOCATION = "test.txt";
  final static int LABEL0 = 4;
  final static int LABEL1 = 7;
  final static int NUM_HIDDEN_LAYER = 1;
  final static int NUM_FEATURE = 784; // MNIST
  final static int[] NUM_UNITS = {392, 1}; // last entry is the number of output
  final static long RANDOM_SEED = 2006031213;
  static ArrayList<NeuralNetworkLayer> bestLayers;
  static double bestLoss = Double.POSITIVE_INFINITY;
  static int bestEpoch;

  /**
   * main method of NeuralNetwork Class
   *
   * @param args Command Line Arguments (CLAs)
   *             0. Question 5 save file name (saveFirstLayerWeightAndBias)
   *             1. Question 6 save file name (saveSecondLayerWeightAndBias)
   *             2. Question 7 save file name (saveOutputActivationTest)
   *             3. Question 8 save file name (saveOutputPredictionTest)
   *             4. Question 9 save file name (saveMostUncertainTestFeature)
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
    System.out.println("Finish Loading Data\n");

    // Initialize Layers with random weights and bias ([-1, 1])
    Random random = new Random(RANDOM_SEED);
    layers = new ArrayList<>(NUM_HIDDEN_LAYER + 1); // hidden layers + output layer
    layers.add(0, new NeuralNetworkLayer(NUM_FEATURE, NUM_UNITS[0], random, -1, 1));
    for(int layerIndex = 1; layerIndex < NUM_HIDDEN_LAYER + 1; layerIndex++) {
      layers.add(layerIndex, new NeuralNetworkLayer(NUM_UNITS[layerIndex - 1], NUM_UNITS[layerIndex], random, -1, 1));
    }

    // Initialize bestLayer with null
    bestLayers = new ArrayList<>(NUM_HIDDEN_LAYER + 1);
    for(int i = 0; i < NUM_HIDDEN_LAYER + 1; i++) {
      bestLayers.add(i, null);
    }

    // Calculate activations of each layer
    ArrayList<ArrayList<Double[]>> activations = new ArrayList<>(NUM_HIDDEN_LAYER + 1);
    activations.add(0, calculateActivation(dataset.getTrainingFeatures(),
        NUM_UNITS[0], layers.get(0))); // first layer (from feature to 1st hidden)
    for(int layerIndex = 1; layerIndex < NUM_HIDDEN_LAYER + 1; layerIndex++) { // for the other layers
      activations.add(layerIndex, calculateActivation(activations.get(layerIndex - 1),
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
        System.out.println("Best Epoch: " + bestEpoch);
        logWriter.write("Best Epoch: " + bestEpoch + "\n");
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

    // Test (Only support one hidden layer case)
    ArrayList<ArrayList<Double[]>> testActivations = new ArrayList<>(NUM_HIDDEN_LAYER + 1);
    testActivations.add(0, calculateActivation(dataset.getTestingFeatures(), NUM_UNITS[0], bestLayers.get(0)));
    testActivations.add(1, calculateActivation(testActivations.get(0), NUM_UNITS[1], bestLayers.get(1)));
    // Calculate prediction
    int[] testPrediction = new int[dataset.getTestingFeatures().size()];
    for(int dataIndex = 0; dataIndex < dataset.getTestingFeatures().size(); dataIndex++) {
      if(testActivations.get(1).get(dataIndex)[0] < 0.5) {
        testPrediction[dataIndex] = 0;
      } else {
        testPrediction[dataIndex] = 1;
      }
    }
    // Get most uncertain dataIndex
    int minIndex = 0;
    double prevMin = 1.0;
    for(int dataIndex = 0; dataIndex < dataset.getTestingFeatures().size(); dataIndex++) {
      if(prevMin > Math.abs(testActivations.get(1).get(dataIndex)[0] - 0.5)) {
        minIndex = dataIndex;
        prevMin = Math.abs(testActivations.get(1).get(dataIndex)[0] - 0.5);
      }
    }

    // Questions
    saveFirstLayerWeightAndBias(args[0]); // Question 5
    saveSecondLayerWeightAndBias(args[1]); // Question 6
    saveOutputActivationTest(args[2], testActivations.get(1)); // Question 7
    saveOutputPredictionTest(args[3], testPrediction); // Question 8
    saveMostUncertainTestFeature(args[4], minIndex); // Question 9
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
    // Frequently used variables
    int numDataEntries = prevActivation.size();

    ArrayList<Double[]> activation = new ArrayList<>(numDataEntries); // place to store activation
    // retrieve weights and bias
    Double[][] weights = layer.getWeights();
    Double[] bias = layer.getBias();

    for(int dataIndex = 0; dataIndex < numDataEntries; dataIndex++) { // for all data entry
      activation.add(dataIndex, new Double[numUnits]); // set the Double array storing activation

      // Frequently using data
      Double[] prevActivationAtData = prevActivation.get(dataIndex);
      Double[] currentActivation = activation.get(dataIndex);

      // calculate activation for each hidden Units
      for(int hiddenUnitIndex = 0; hiddenUnitIndex < numUnits; hiddenUnitIndex++) {
        // calculate weighted sum
        double weightedSum = 0.0;
        for(int prevActivationIndex = 0; prevActivationIndex < prevActivationAtData.length; prevActivationIndex++) {
          weightedSum += weights[prevActivationIndex][hiddenUnitIndex] * prevActivationAtData[prevActivationIndex];
        }
        weightedSum += bias[hiddenUnitIndex];

        // calculate activation based on the weighted sum
        currentActivation[hiddenUnitIndex] = NeuralNetworkFunction.logisticSigmoid(weightedSum);
      }
    }

    return activation;
  }

  /**
   * Private helper method to update Weights and Bias of the neural network
   * (Only support one hidden layer case)
   *
   * @param activations previously calculated activation
   *                    Outer ArrayList - Indicates each layer
   *                    Inner ArrayList - Indicates data points
   *                    Double[] Array - The activation of the unit
   */
  private static void updateWeightsAndBias(ArrayList<ArrayList<Double[]>> activations) {
    // variables and constant used frequently
    int dataLength = activations.get(0).size();
    ArrayList<Integer> trueLabel = dataset.getTrainingLabels();
    ArrayList<Double[]> outputLayerActivation = activations.get(1);

    // Calculate derivative of Loss function w.r.t. net input of output layer
    // dL/dz_output
    double[] derivLoss = new double[dataLength];
    for(int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
      derivLoss[dataIndex] = outputLayerActivation.get(dataIndex)[0] - trueLabel.get(dataIndex);
    }

    // variables and constant used frequently
    Double[][] hiddenLayerWeight = layers.get(0).getWeights();
    Double[] hiddenLayerBias = layers.get(0).getBias();
    Double[][] outputLayerWeight = layers.get(1).getWeights();
    Double[] outputLayerBias = layers.get(1).getBias();

    // Update Hidden Layer (first layer)
    for(int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
      // variables and constant used frequently
      Double[] activation = activations.get(0).get(dataIndex);
      Double[] features = dataset.getTrainingFeatures().get(dataIndex);

      for(int hiddenUnitIndex = 0; hiddenUnitIndex < NUM_UNITS[0]; hiddenUnitIndex++) {
        // update bias
        hiddenLayerBias[hiddenUnitIndex] -= LEARNING_RATE
            * (derivLoss[dataIndex] * outputLayerWeight[hiddenUnitIndex][0]
            * NeuralNetworkFunction.diffLogisticSigmoid(activation[hiddenUnitIndex]) * 1.0);

        for(int featureIndex = 0; featureIndex < NUM_FEATURE; featureIndex++) {
          // Update weight
          hiddenLayerWeight[featureIndex][hiddenUnitIndex] -= LEARNING_RATE
              * (derivLoss[dataIndex] * outputLayerWeight[hiddenUnitIndex][0]
              * NeuralNetworkFunction.diffLogisticSigmoid(activation[hiddenUnitIndex]) * features[featureIndex]);
        }
      }
    }

    // Update output layer
    for(int dataIndex = 0; dataIndex < dataLength; dataIndex++) {
      // update bias
      outputLayerBias[0] -= LEARNING_RATE * (derivLoss[dataIndex] * 1.0);

      // variables and constant used frequently
      Double[] activation = activations.get(0).get(dataIndex);

      for(int weightIndex = 0; weightIndex < NUM_UNITS[0]; weightIndex++) {
        // update documents
        outputLayerWeight[weightIndex][0] -= LEARNING_RATE * (derivLoss[dataIndex] * activation[weightIndex]);
      }
    }
  }

  /**
   * Private helper method to check convergence by calculating loss function
   * (Only support one hidden layer case)
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
          output.get(dataIndex)[0], dataset.getTrainingLabels().get(dataIndex));
    }

    if(currentLoss <= bestLoss) { // new best found, save the weights
      bestLoss = currentLoss;
      for(int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        bestLayers.set(layerIndex, layers.get(layerIndex).getDuplicated());
      }
      bestEpoch = epoch;

      // logging
      System.out.println("Loss: " + currentLoss + ", New Best Found!!");
      logWriter.write("Loss: " + currentLoss + ", New Best Found!!\n");
    } else {
      // logging
      System.out.println("Loss: " + currentLoss);
      logWriter.write("Loss: " + currentLoss + "\n");
    }

    // Check for convergence
    return Math.abs(previousLoss - currentLoss) < EPSILON;
  }

  /**
   * Method to handle Question 5 - save first layer weights and bias
   *
   * @param filename filename to save the answer
   * @throws IOException while writing the file, may throw IOException
   */
  private static void saveFirstLayerWeightAndBias(String filename) throws IOException {
    // Create File Writer
    File destination = new File(filename);
    FileWriter fileWriter = new FileWriter(destination);

    // Retrieve the Weight and Bias
    Double[] bias = bestLayers.get(0).getBias();
    Double[][] weights = bestLayers.get(0).getWeights();

    // Write weights
    for(int featureIndex = 0; featureIndex < NUM_FEATURE; featureIndex++) {
      for(int outputIndex = 0; outputIndex < NUM_UNITS[0] - 1; outputIndex++) {
        fileWriter.write(String.format("%.4f", weights[featureIndex][outputIndex]) + ",");
      }
      fileWriter.write(String.format("%.4f", weights[featureIndex][NUM_UNITS[0] - 1]) + "\n");
    }

    // Write bias
    for(int outputIndex = 0; outputIndex < NUM_UNITS[0] - 1; outputIndex++) {
      fileWriter.write(String.format("%.4f", bias[outputIndex]) + ",");
    }
    fileWriter.write(String.format("%.4f", bias[NUM_UNITS[0] - 1]));

    // Flush the buffer
    fileWriter.flush();
    fileWriter.close();
  }

  /**
   * Method to handle Question 6 - save first layer weights and bias
   *
   * @param filename filename to save the answer
   * @throws IOException while writing the file, may throw IOException
   */
  private static void saveSecondLayerWeightAndBias(String filename) throws IOException {
    // Create File Writer
    File destination = new File(filename);
    FileWriter fileWriter = new FileWriter(destination);

    // Retrieve the Weight and Bias
    Double[] bias = bestLayers.get(1).getBias();
    Double[][] weights = bestLayers.get(1).getWeights();

    // write weights
    for(int activationIndex = 0; activationIndex < NUM_UNITS[0]; activationIndex++) {
      fileWriter.write(String.format("%.4f", weights[activationIndex][0]) + ",");
    }
    // write bias
    fileWriter.write(String.format("%.4f", bias[0]));

    // Flush the buffer
    fileWriter.flush();
    fileWriter.close();
  }

  /**
   * Method to handle Question 7 - save output layer activation
   *
   * @param filename         filename to save the answer
   * @param outputActivation ArrayList of Double array contains output layer's activation
   * @throws IOException while writing the file, may throw IOException
   */
  private static void saveOutputActivationTest(String filename, ArrayList<Double[]> outputActivation)
      throws IOException {
    // Create File Writer
    File destination = new File(filename);
    FileWriter fileWriter = new FileWriter(destination);

    // Write Activations
    for(int dataIndex = 0; dataIndex < outputActivation.size() - 1; dataIndex++) {
      fileWriter.write(String.format("%.2f", outputActivation.get(dataIndex)[0]) + ",");
    }
    fileWriter.write(String.format("%.2f", outputActivation.get(outputActivation.size() - 1)[0]));

    // Flush the buffer
    fileWriter.flush();
    fileWriter.close();
  }

  /**
   * Method to handle Question 8 - save output layer prediction
   *
   * @param filename         filename to save the answer
   * @param outputPrediction int array with
   * @throws IOException while writing the file, may throw IOException
   */
  private static void saveOutputPredictionTest(String filename, int[] outputPrediction) throws IOException {
    // Create File Writer
    File destination = new File(filename);
    FileWriter fileWriter = new FileWriter(destination);

    // Write Prediction
    for(int dataIndex = 0; dataIndex < outputPrediction.length - 1; dataIndex++) {
      fileWriter.write(String.format("%d", outputPrediction[dataIndex]) + ",");
    }
    fileWriter.write(String.format("%d", outputPrediction[outputPrediction.length - 1]));

    // Flush the buffer
    fileWriter.flush();
    fileWriter.close();
  }

  /**
   * Method to handle Question 9 - save feature of test example that make most ambiguous prediction
   *
   * @param filename filename to save the answer
   * @param minIndex index of Test cases that making most ambiguous prediction
   * @throws IOException while writing the file, may throw IOException
   */
  private static void saveMostUncertainTestFeature(String filename, int minIndex) throws IOException {
    // Create File Writer
    File destination = new File(filename);
    FileWriter fileWriter = new FileWriter(destination);

    // Retrieve the test example
    Double[] testExample = dataset.getTestingFeatures().get(minIndex);

    // Write the example feature
    for(int featureIndex = 0; featureIndex < NUM_FEATURE - 1; featureIndex++) {
      fileWriter.write(String.format("%.2f", testExample[featureIndex]) + ",");
    }
    fileWriter.write(String.format("%.2f", testExample[NUM_FEATURE - 1]));

    // Flush the buffer
    fileWriter.flush();
    fileWriter.close();
  }

}
