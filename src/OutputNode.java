/**
 * This class implements an output node of a Feedforward Neural Network
 * 
 * @author plin
 */

public class OutputNode 
{
    double[] weights; // For n hidden nodes, w1...wN
    double w0; // Bias
    double[] oldWeights; // Previous set of weights
    double oldW0; // Previous bias
    double output; // Sigmoid output
    double activation; // Weighted sum
    double eta = 0.2; // Learning rate
    double dEdY; // Derivative of the error with respect to the sigmoid
    double dYdZ; // Derivative of the sigmoid with respect to the weighted sum
    String classVal; // Class value of the output node
    
    public OutputNode(int numHidden, String classVal, double eta)
    {
        weights = new double[numHidden];
        this.classVal = classVal;
        this.eta = eta;
        
        // Initialize weights in the range [-0.01, 0.01]
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] = Math.random()*0.02 - 0.01;
        }
        w0 = Math.random()*0.02 - 0.01;
        
        oldWeights = weights;
        oldW0 = w0;
    }
    
    /**
     * This method calculates the weighted sum of the previous layer (net 
     * output)
     * 
     * @param z is the input array or an array of sigmoid outputs from the 
     * previous hidden layer
     */
    public void activate(String[] z)
    {
        double weightedSum = 0;
        for (int i = 0; i < z.length; i++)
        {
            weightedSum += weights[i] * Double.parseDouble(z[i]);
        }
        activation = weightedSum + w0;
    }
    
    /**
     * This method calculates the sigmoid output from the net output
     */
    public void sigmoid()
    {
        output = 1.0 / (1 + Math.exp(-activation));
    }
    
    /**
     * This method update the weights using gradient descent
     * 
     * @param z is the input array or an array of sigmoid outputs from the 
     * previous hidden layer 
     * @param expected is the expected sigmoid output from the node
     */
    public void updateWeights(String[] z, double expected)
    {
        oldWeights = weights;
        oldW0 = w0;
        dEdY = (output - expected);
        dYdZ = output * (1 - output);
        
        for (int i = 0; i < weights.length; i++)
        {
            weights[i] -= eta * dEdY * dYdZ * Double.parseDouble(z[i]);
        }
        w0 -= eta * dEdY * dYdZ;
    }
}
