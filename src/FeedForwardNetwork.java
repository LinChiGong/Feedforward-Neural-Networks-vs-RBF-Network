/**
 * This class implements the Feedforward Neural Network Algorithm for
 * classification. Backpropagation is used for training. The default structure
 * contains one hidden layer with n hidden nodes where n = number of classes. 
 * The dataset used to fit the model should be normalized beforehand
 * 
 * @author Winston Lin
 */

import java.util.*;

public class FeedForwardNetwork 
{
    ArrayList<HiddenNode[]> hiddenLayers = 
            new ArrayList<HiddenNode[]>(); // Stores all hidden layers
    OutputNode[] outputLayer; // Stores the output nodes
    int numHiddenLayers = 1;  // Number of hidden layers
    int numHiddenNodes = -1;  // Number of hidden nodes in each hidden layer
    double threshold = 0.75;  // Gradient descent convergence threshold
    int iteration = 10000;    // Gradient descent convergence iteration
    double eta = 0.2; // Learning rate
    
    public FeedForwardNetwork()
    {// Initialize with default structure and convergence criteria
    }
    
    public FeedForwardNetwork(int numHiddenLayers, int numHiddenNodes, 
            double threshold, int iteration, double eta)
    {// Initialize with specified structure and convergence criteria
        this.numHiddenLayers = numHiddenLayers;
        this.numHiddenNodes = numHiddenNodes;
        this.threshold = threshold;
        this.iteration = iteration;
        this.eta = eta;
    }
    
    /**
     * This method trains the Feedforward Network model using the training set
     * 
     * @param X is a matrix of attributes of the training instances
     * @param y is an array of targets of the training instances
     */
    public void fit(ArrayList<String[]> X, ArrayList<String> y)
    {
        // Initialize hidden nodes and output nodes
        ArrayList<String> classes = getUnique(y);
        if (numHiddenNodes == -1)
        {// Default number of hidden nodes = Number of classes
            numHiddenNodes = classes.size();
        }
        for (int i = 0; i < numHiddenLayers; i++)
        {
            HiddenNode[] hiddenLayer = new HiddenNode[numHiddenNodes];
            for (int j = 0; j < hiddenLayer.length; j++)
            {
                if (i == 0)
                {// First hidden layer: number of weights = number of features
                    hiddenLayer[j] = new HiddenNode(X.get(0).length, eta);
                }
                else
                {// Others: number of weights = number of hidden nodes
                    hiddenLayer[j] = new HiddenNode(numHiddenNodes, eta);
                }
            }
            hiddenLayers.add(hiddenLayer);
        }
        outputLayer = new OutputNode[classes.size()];
        for (int i = 0; i < classes.size(); i++)
        {
            if (numHiddenLayers == 0)
            {// No hidden layers: number of weights = number of features
                outputLayer[i]= new OutputNode(X.get(0).length,classes.get(i), eta);
            }
            else
            {// 1+ hidden layers: number of weights = number of hidden nodes
                outputLayer[i] = new OutputNode(numHiddenNodes,classes.get(i), eta);
            }
        }
        
        // Perform gradient descent
        double performance = 0;
        int iter = 0;
        while (iter < iteration && performance < threshold)
        {
            // Select training instances in random order
            ArrayList<Integer> indexArray = new ArrayList<Integer>();
            for (int i = 0; i < X.size(); i++)
            {
                indexArray.add(i);
            }
            Collections.shuffle(indexArray);
            for (int i = 0; i < indexArray.size(); i++)
            {
                String[] x = X.get(indexArray.get(i));
                
                // Forward propagate an input instance to a network output
                if (numHiddenLayers == 0)
                {// Directly update weights when there is no hidden layers
                    for (int k = 0; k < outputLayer.length; k++)
                    {
                        outputLayer[k].activate(x);
                        outputLayer[k].sigmoid();
                        if (outputLayer[k].classVal.equals(
                                y.get(indexArray.get(i))))
                        {
                            outputLayer[k].updateWeights(x, 1);
                        }
                        else
                        {
                            outputLayer[k].updateWeights(x, 0);
                        }
                    }
                }
                else
                {
                    String[] h = new String[numHiddenNodes];
                    for (int j = 0; j < hiddenLayers.size() + 1; j++)
                    {
                        if (j == 0)
                        {// First hidden layer takes inputs from training set
                            for (int k = 0; k < numHiddenNodes; k++)
                            {
                                hiddenLayers.get(0)[k].activate(x);
                                hiddenLayers.get(0)[k].sigmoid();
                                h[k] = 
                                Double.toString(hiddenLayers.get(0)[k].output);
                            }
                        }
                        else if (j < hiddenLayers.size())
                        {// Other hidden layers take inputs from previous layer
                            for (int k = 0; k < numHiddenNodes; k++)
                            {
                                hiddenLayers.get(j)[k].activate(h);
                                hiddenLayers.get(j)[k].sigmoid();
                                h[k] = 
                                Double.toString(hiddenLayers.get(j)[k].output);
                            }
                        }
                        else
                        {// Calculate output and update weights
                            for (int k = 0; k < outputLayer.length; k++)
                            {
                                outputLayer[k].activate(h);
                                outputLayer[k].sigmoid();
                                if (outputLayer[k].classVal.equals(
                                        y.get(indexArray.get(i))))
                                {
                                    outputLayer[k].updateWeights(h, 1);
                                }
                                else
                                {
                                    outputLayer[k].updateWeights(h, 0);
                                }
                            }
                        }
                    }
                }
                
                // Back propagate to update weights of the hidden nodes
                if (hiddenLayers.size() == 1)
                {
                    for (int k = 0; k < hiddenLayers.get(0).length; k++)
                    {
                        double dEdY = 0; // dEdY of each hidden node
                        for (int l = 0; l < outputLayer.length; l++)
                        {
                            dEdY += outputLayer[l].dEdY 
                                    * outputLayer[l].dYdZ
                                    * outputLayer[l].oldWeights[k];
                        }
                        
                        for (int l = 0; l < x.length; l++)
                        {
                            hiddenLayers.get(0)[k].calculateDerivatives(x[l]);
                            hiddenLayers.get(0)[k].dEdW = dEdY 
                                    * hiddenLayers.get(0)[k].dYdZ 
                                    * hiddenLayers.get(0)[k].dZdW;
                            hiddenLayers.get(0)[k].weights[l] -= 
                                      hiddenLayers.get(0)[k].eta 
                                    * hiddenLayers.get(0)[k].dEdW;
                            hiddenLayers.get(0)[k].w0 -= dEdY
                                    * hiddenLayers.get(0)[k].eta 
                                    * hiddenLayers.get(0)[k].dYdZ;
                        }
                    }
                }
                else
                {
                    for (int j = hiddenLayers.size() - 1; j >= 0; j--)
                    {
                        if (j == hiddenLayers.size() - 1)
                        {// Last hidden layer: back propagate from output layer
                            for (int k = 0;k < hiddenLayers.get(j).length;k++)
                            {
                                double dEdY = 0;
                                for (int l = 0; l < outputLayer.length; l++)
                                {
                                    dEdY += outputLayer[l].dEdY 
                                            * outputLayer[l].dYdZ
                                            * outputLayer[l].oldWeights[k];
                                }
                                hiddenLayers.get(j)[k].oldWeights = 
                                        hiddenLayers.get(j)[k].weights;
                                hiddenLayers.get(j)[k].oldW0 = 
                                        hiddenLayers.get(j)[k].w0;
                                hiddenLayers.get(j)[k].dEdY = dEdY;
                                
                                for (int l = 0; l < numHiddenNodes; l++)
                                {
                                    // Inputs are outputs from previous layer
                                    hiddenLayers.get(j)[k].
                                    calculateDerivatives(Double.toString(
                                           hiddenLayers.get(j - 1)[l].output));
                                    hiddenLayers.get(j)[k].dEdW = dEdY 
                                            * hiddenLayers.get(j)[k].dYdZ 
                                            * hiddenLayers.get(j)[k].dZdW;
                                    hiddenLayers.get(j)[k].weights[l] -= 
                                              hiddenLayers.get(j)[k].eta 
                                            * hiddenLayers.get(j)[k].dEdW;
                                    hiddenLayers.get(j)[k].w0 -= dEdY
                                            * hiddenLayers.get(j)[k].eta 
                                            * hiddenLayers.get(j)[k].dYdZ;
                                }
                            }
                        }
                        else
                        {// For others, back propagate from next hidden layer
                            for (int k = 0;k < hiddenLayers.get(j).length;k++)
                            {
                                double dEdY = 0;
                                for (int l = 0; l < numHiddenNodes; l++)
                                {
                                    dEdY += hiddenLayers.get(j+1)[l].dEdY
                                            * hiddenLayers.get(j+1)[l].dYdZ
                                            * hiddenLayers.get(j+1)[l].
                                            oldWeights[k];
                                }
                                hiddenLayers.get(j)[k].oldWeights = 
                                        hiddenLayers.get(j)[k].weights;
                                hiddenLayers.get(j)[k].oldW0 = 
                                        hiddenLayers.get(j)[k].w0;
                                hiddenLayers.get(j)[k].dEdY = dEdY;
                                
                                if (j == 0)
                                {// First layer, update weights using inputs
                                    for (int l = 0; l < x.length; l++)
                                    {
                                        hiddenLayers.get(0)[k].
                                        calculateDerivatives(x[l]);
                                        hiddenLayers.get(0)[k].dEdW = dEdY 
                                                * hiddenLayers.get(0)[k].dYdZ
                                                * hiddenLayers.get(0)[k].dZdW;
                                        hiddenLayers.get(0)[k].weights[l] -= 
                                                  hiddenLayers.get(0)[k].eta 
                                                * hiddenLayers.get(0)[k].dEdW;
                                        hiddenLayers.get(0)[k].w0 -= dEdY
                                                * hiddenLayers.get(0)[k].eta 
                                                * hiddenLayers.get(0)[k].dYdZ;
                                    }
                                }
                                else
                                {// Others, update weights using previous layer
                                    for (int l = 0; l < numHiddenNodes; l++)
                                    {
                                        hiddenLayers.get(j)[k].
                                        calculateDerivatives(Double.toString(
                                           hiddenLayers.get(j - 1)[l].output));
                                        hiddenLayers.get(j)[k].dEdW = dEdY 
                                                * hiddenLayers.get(j)[k].dYdZ 
                                                * hiddenLayers.get(j)[k].dZdW;
                                        hiddenLayers.get(j)[k].weights[l] -= 
                                                  hiddenLayers.get(j)[k].eta 
                                                * hiddenLayers.get(j)[k].dEdW;
                                        hiddenLayers.get(j)[k].w0 -= dEdY
                                                * hiddenLayers.get(j)[k].eta 
                                                * hiddenLayers.get(j)[k].dYdZ;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            performance = trainingPerformance(X, y);
            iter++;
            
            // For demonstration purpose
            System.out.println("adjusting weights ... current performance: "
                                + performance);
        }
    }
    
    /**
     * This method calculates the training accuracy during each iteration of
     * Backpropagation
     * 
     * @param X is a matrix of attributes of the training instances
     * @param y is an array of targets of the training instances
     * @return the training accuracy
     */
    public double trainingPerformance(ArrayList<String[]>X, ArrayList<String>y)
    {
        double performance = 0;

        for (int i = 0; i < X.size(); i++)
        {
            String classVal = "";
            double bestScore = -1;
            
            // Forward propagate input to a network output
            if (numHiddenLayers == 0)
            {// Case when there is no hidden layer: Select the best output
                for (int k = 0; k < outputLayer.length; k++)
                {
                    outputLayer[k].activate(X.get(i));
                    outputLayer[k].sigmoid();
                    
                    if (outputLayer[k].output > bestScore)
                    {
                        bestScore = outputLayer[k].output;
                        classVal = outputLayer[k].classVal;
                    }
                }
            }
            else
            {
                String[] h = new String[numHiddenNodes];
                for (int j = 0; j < hiddenLayers.size() + 1; j++)
                {
                    if (j == 0)
                    {// First hidden layer takes inputs from training set
                        for (int k = 0; k < numHiddenNodes; k++)
                        {
                            hiddenLayers.get(0)[k].activate(X.get(i));
                            hiddenLayers.get(0)[k].sigmoid();
                            h[k] = 
                                Double.toString(hiddenLayers.get(0)[k].output);
                        }
                    }
                    else if (j < hiddenLayers.size())
                    {// Other hidden layers takes inputs from previous layer
                        for (int k = 0; k < numHiddenNodes; k++)
                        {
                            hiddenLayers.get(j)[k].activate(h);
                            hiddenLayers.get(j)[k].sigmoid();
                            h[k] = 
                                Double.toString(hiddenLayers.get(j)[k].output);
                        }
                    }
                    else
                    {// Calculate output for each output node & select the best
                        for (int k = 0; k < outputLayer.length; k++)
                        {
                            outputLayer[k].activate(h);
                            outputLayer[k].sigmoid();
                            
                            if (outputLayer[k].output > bestScore)
                            {
                                bestScore = outputLayer[k].output;
                                classVal = outputLayer[k].classVal;
                            }
                        }
                    }
                }
            }
            
            // Compare the prediction to the true class value
            if (classVal.equals(y.get(i)))
            {
                performance++;
            }
        }
        performance /= X.size();
        
        return performance;
    }
    
    /**
     * This method makes prediction on the test set using the trained model
     * 
     * @param X is the test set
     * @return an array of predictions
     */
    public ArrayList<String> predict(ArrayList<String[]> X)
    {
        ArrayList<String> ypred = new ArrayList<String>();

        for (int i = 0; i < X.size(); i++)
        {
            String classVal = "";
            double bestScore = -1;
            
            // Forward propagate input to a network output
            if (numHiddenLayers == 0)
            {// Case when there is no hidden layer: Select the best output
                for (int k = 0; k < outputLayer.length; k++)
                {
                    outputLayer[k].activate(X.get(i));
                    outputLayer[k].sigmoid();
                    
                    if (outputLayer[k].output > bestScore)
                    {
                        bestScore = outputLayer[k].output;
                        classVal = outputLayer[k].classVal;
                    }
                }
            }
            else
            {
                String[] h = new String[numHiddenNodes];
                for (int j = 0; j < hiddenLayers.size() + 1; j++)
                {
                    if (j == 0)
                    {// First hidden layer takes inputs from training set
                        for (int k = 0; k < numHiddenNodes; k++)
                        {
                            hiddenLayers.get(0)[k].activate(X.get(i));
                            hiddenLayers.get(0)[k].sigmoid();
                            h[k] = 
                                Double.toString(hiddenLayers.get(0)[k].output);
                        }
                    }
                    else if (j < hiddenLayers.size())
                    {// Other hidden layers takes inputs from previous layer
                        for (int k = 0; k < numHiddenNodes; k++)
                        {
                            hiddenLayers.get(j)[k].activate(h);
                            hiddenLayers.get(j)[k].sigmoid();
                            h[k] = 
                                Double.toString(hiddenLayers.get(j)[k].output);
                        }
                    }
                    else
                    {// Calculate output for each output node & select the best
                        for (int k = 0; k < outputLayer.length; k++)
                        {
                            outputLayer[k].activate(h);
                            outputLayer[k].sigmoid();
                            
                            if (outputLayer[k].output > bestScore)
                            {
                                bestScore = outputLayer[k].output;
                                classVal = outputLayer[k].classVal;
                            }
                        }
                    }
                }
            }
            
            // Predict the class with the highest score
            ypred.add(classVal);
        }
        
        return ypred;
    }
    
    /**
     * This method finds all unique values in an array
     * 
     * @param y is the array of interest
     * @return a new array without redundant values
     */
    public ArrayList<String> getUnique(ArrayList<String> y)
    {
        ArrayList<String> uniqueValues = new ArrayList<String>();

        for(int i = 0; i < y.size(); i++)
        {
            if(!uniqueValues.contains(y.get(i)))
            {
                uniqueValues.add(y.get(i));
            }
        }

       return uniqueValues;
    }
}
