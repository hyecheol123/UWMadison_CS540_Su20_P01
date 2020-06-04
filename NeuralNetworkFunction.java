///////////////////////////////// FILE HEADER /////////////////////////////////
//
// Title:           UWMadison_CS540_Su20_P01
// This File:       NeuralNetworkFunction.java
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

/**
 * Private inner class having static functions that will be used for gradient descent
 */
class NeuralNetworkFunction {
  /**
   * Calculate logistic sigmoid function
   * @param weightedSum input for the function
   * @return return of logistic sigmoid function
   */
  public static double logisticSigmoid(double weightedSum) {
    return 1.0 / (1.0 + Math.exp(-1.0 * weightedSum));
  }

  /**
   * Differentiation of Logistic Sigmoid, g'(x) = g(x) * (1 - g(x))
   *
   * @param activation output of logistic sigmoid function (previously calculated)
   * @return differentiation of logistic sigmoid
   */
  public static double diffLogisticSigmoid(double activation) {
    return activation * (1.0 - activation);
  }

  /**
   * calculate binary cross entropy for each data point (prediction)
   * C = -{y * log(a) + (1 - y) * log(1 - a)}
   *
   * @param prediction predicted output (activation)
   * @param label      truth label
   * @return calculation of binary cross entropy
   */
  public static double binaryCrossEntropy(double prediction, int label) {
    if(label == 1) {
      if(prediction < 0.0001) { // To prevent NaN and Inf, for the cases possibly cause Inf, just add a large number
        return 100;
      } else {
        return (-1.0) * Math.log(prediction);
      }
    } else { // when label is 0
      if(prediction > 0.9999) { // To prevent NaN and Inf, for the cases possibly cause Inf, just add a large number
        return 100;
      } else {
        return (-1.0) * Math.log(1.0 - prediction);
      }
    }
  }
}