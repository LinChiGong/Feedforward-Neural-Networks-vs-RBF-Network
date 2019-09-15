/**
 * This class implements a hidden node of a Feedforward Neural Network
 * 
 * @author plin
 */

public class HiddenNode 
{
    double[] weights; // For n input nodes, w1...wN 
    double w0; // Bias
    double[] oldWeights; // Previous set of weights
    double oldW0; // Previous bias
    double output; // Sigmoid output
    double activation; // Weighted sum
    double dEdY; // Derivative of the error with respect to the sigmoid
    double dYdZ; // Derivative of the sigmoid with respect to the weighted sum
    double dZdW; // Derivative of the weighted sum with respect to the weight
    double dEdW; // Derivative of the error with respect to the weight
    double eta = 0.2; // Learning rate
    
    public HiddenNode(int numInput, double eta)
    {
        weights = new double[numInput];
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
     * @param x is the input array or an array of sigmoid outputs from the 
     * previous hidden layer
     */
    public void activate(String[] x)
    {
        double weightedSum = 0;
        for (int i = 0; i < x.length; i++)
        {
            weightedSum += weights[i] * Double.parseDouble(x[i]);
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
     * This method calculates the derivatives needed for weight update
     * 
     * @param xValue is the output of one node in the previous layer
     */
    public void calculateDerivatives(String xValue)
    {
        dYdZ = output * (1-output);
        dZdW = Double.parseDouble(xValue);
    }
}
