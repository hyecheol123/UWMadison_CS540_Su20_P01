///////////////////////////////// FILE HEADER /////////////////////////////////
//
// Title:           UWMadison_CS540_Su20_P01
// This File:       NeuralNetworkLayer.java
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

import java.util.Random;

/**
 * Class for each neural network's layer
 */
class NeuralNetworkLayer {
  // Class Variables
  private Double[][] weights;
  private Double[] bias;

  /**
   * Initialize NeuralNetworkLayer, with weights and bias initialized to 0.0
   *
   * @param input input dimension
   * @param output output dimension
   */
  NeuralNetworkLayer(int input, int output) {
    weights = new Double[input][output];
    bias = new Double[output];
  }

  /**
   * Initialize Neural Network, with weights and bias initialized to the random number distributed within [lower, upper]
   *
   * @param input input dimension
   * @param output output dimension
   * @param random Random instance that will be used to initialize weights and bias
   * @param lower lower bound of weight and bias
   * @param upper upper bound of weight and bias
   */
  NeuralNetworkLayer(int input, int output, Random random, double lower, double upper) {
    this(input, output);

    // Initialize weights and bias randomly within [-1, 1]
    for(int j = 0; j < output; j++) {
      for(int i = 0; i < input; i++) {
        weights[i][j] = random.nextDouble() * (upper - lower) + lower;
      }
      bias[j] = random.nextDouble() * (upper - lower) + lower;
    }
  }

  /**
   * Accessor of the weights
   *
   * @return 2D Double matrix of weights
   */
  public Double[][] getWeights() {
    return weights;
  }

  /**
   * Accessor of bias
   *
   * @return 1D Double array of bias
   */
  public Double[] getBias() {
    return bias;
  }
}