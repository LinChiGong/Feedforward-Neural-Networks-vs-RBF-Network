/**
 * This class implements the RBF Network Algorithm for classification. The
 * dataset used to fit the model should be normalized beforehand. The inner
 * class "Node" implements the hidden node
 * 
 * @author Winston Lin
 */

import java.util.*;

public class RBFNetwork
{
    class Node
    {
        String[] center;   // Center of the hidden node
        double sigma = 0;  // Spread of the hidden node
        double output = 0; // Output of the hidden node
        ArrayList<Double> weights = new ArrayList<Double>(); // Stores weights
        
        public Node(String[] center)
        {
            this.center = center;
        }
        
        /**
         * This class calculates the RBF value for a query point
         * 
         * @param X is the query point
         */
        public void RBF(String[] X)
        {
            double distance = 0;
            for (int i = 0; i < X.length; i++)
            {
                distance += Math.pow((Double.parseDouble(X[i]) 
                        - Double.parseDouble(center[i])), 2);
            }
            output = Math.exp(((-1.0) / (2.0 * sigma * sigma)) * distance);
        }
        
        /**
         * This class sets the value of spread
         * 
         * @param sigma = spread
         */
        public void setSigma(double sigma)
        {
            this.sigma = sigma;
        }
    }
    
    ArrayList<String> classes = new ArrayList<String>(); // Class names
    ArrayList<Node> nodes = new ArrayList<Node>(); // Hidden nodes
    int numHidden = 10; // Number of hidden nodes
    double threshold = 0.95; // Gradient descent convergence threshold
    int iteration = 1000;   // Gradient descent convergence iteration
    double eta = 20; // Learning rate
    
    public RBFNetwork()
    {// Initialize with default number of hidden nodes & convergence criteria
    }
    
    public RBFNetwork(int numHidden, double threshold, int iteration, double eta)
    {// Initialize with specified number of hidden nodes & convergence criteria
        this.numHidden = numHidden;
        this.threshold = threshold;
        this.iteration = iteration; 
        this.eta = eta;
    }
 
    /**
     * This method trains the RBF Network model using the training set
     * 
     * @param X is a matrix of attributes of the training instances
     * @param y is an array of targets of the training instances
     */
    public void fit(ArrayList<String[]> X, ArrayList<String> y)
    {
        // Find out all classes
        classes = getUnique(y);
        double[] classOutputs = 
                new double[classes.size()]; // Output for each class
        double[] logisticClassOutputs = 
                new double[classes.size()]; // Logistic output for each class
        
        // Train the RBF classifier
        double n = numHidden;
        for (int i = 0; i < n; i++)
        {// Establish n hidden nodes
            int random = (int) Math.random() * X.size();
            nodes.add(new Node(X.get(random)));
            X.remove(random);
            y.remove(random);
        }
        
        // Calculate spread(sigma) using the max distance among nodes
        double maxDistance = Double.MIN_VALUE;
        for (int i = 0; i < nodes.size() - 1; i++)
        {
            for (int j = i + 1; j < nodes.size(); j++)
            {
                double centerDistance = getDistance(nodes.get(i).center,
                        nodes.get(j).center);
                if (centerDistance > maxDistance)
                {
                    maxDistance = centerDistance;
                }
            }
        }
        double sigma = maxDistance / Math.sqrt(2 * nodes.size());
        for (Node node : nodes)
        {// Set sigma
            node.setSigma(sigma);
        }
        
        for (int i = 0; i < classes.size(); i++)
        {// Initialize weights in the range [-0.01, 0.01]
            for (Node node : nodes)
            {
                node.weights.add(Math.random()*0.02 - 0.01);
            }
        }

        double trainingAccuracy = -1;
        int iter = 0;
        // Continue adjusting the weight until accuracy reaches the threshold
        while (iter < iteration && trainingAccuracy < threshold)
        {
            double correctPrediction = 0; // Number of correct predictions
            for (int i = 0; i < X.size(); i++)
            {
                // Calculate output of each hidden node
                for (Node node : nodes)
                {
                    node.RBF(X.get(i));
                }
                
                // Calculate the output with current weight for each class
                for (int j = 0; j < classes.size(); j++)
                {
                    double classOutput = 0;
                    for (Node node : nodes)
                    {
                        classOutput += (node.weights.get(j) * node.output);
                    }
                    classOutputs[j] = classOutput;
                }
                
                // Transform the outputs using the sigmoid function
                for (int j = 0; j < classes.size(); j++)
                {
                    logisticClassOutputs[j] = 1 / (1 + Math.exp((-1.0) 
                            * classOutputs[j]));
                }
                
                // Prediction would be the class with highest probability
                double maxProbability = -1;
                int finalOutput = 0;
                for (int j = 0; j < classes.size(); j++)
                {
                    if (logisticClassOutputs[j] > maxProbability)
                    {
                        maxProbability = logisticClassOutputs[j];
                        finalOutput = j;
                    }
                }
                if (classes.get(finalOutput).equals(y.get(i)))
                {
                    correctPrediction++;
                }
                
                // Update weights for each class using gradient descent
                for (int j = 0; j < classes.size(); j++)
                {
                    int trueProbability = 0;
                    if (classes.get(j).equals(y.get(i)))
                    {// Weights are updated according to the target class
                        trueProbability = 1;
                    }
                    for (Node node : nodes)
                    {// Increase weight if target class, decrease otherwise
                        double gradient = (trueProbability 
                                - logisticClassOutputs[j]) 
                                * logisticClassOutputs[j] 
                                        * (1 - logisticClassOutputs[j]) 
                                        * node.output;
                        node.weights.set(j, node.weights.get(j) 
                                + eta * gradient);
                    }
                }
            }
            trainingAccuracy = correctPrediction / X.size();  
            iter++;
            
            // For demonstration purpose
            System.out.println("adjusting weights ... current performance: "
            + trainingAccuracy);
        }
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
        double[] classOutputs = 
                new double[classes.size()]; // Output for each class
        double[] logisticClassOutputs = 
                new double[classes.size()]; // Logistic output for each class
        
        // For each test sample, make a prediction about its target label
        for (int i = 0; i < X.size(); i++)
        {
            // Calculate output of each hidden node
            for (Node node : nodes)
            {
                node.RBF(X.get(i));
            }
            
            // Calculate the output for each class
            for (int j = 0; j < classes.size(); j++)
            {
                double classOutput = 0;
                for (Node node : nodes)
                {
                    classOutput += (node.weights.get(j) * node.output);
                }
                classOutputs[j] = classOutput;
            }
            
            // Transform the outputs using the sigmoid function
            for (int j = 0; j < classes.size(); j++)
            {
                logisticClassOutputs[j] = 1 / (1 + Math.exp((-1.0) 
                        * classOutputs[j]));
            }
            
            // Prediction would be the class with highest probability
            double maxProbability = -1;
            int finalOutput = 0;
            for (int j = 0; j < classes.size(); j++)
            {
                if (logisticClassOutputs[j] > maxProbability)
                {
                    maxProbability = logisticClassOutputs[j];
                    finalOutput = j;
                }
            }
            ypred.add(classes.get(finalOutput));
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
    
    /**
     * This method calculates the Euclidean distance between two points
     * 
     * @param x1 is the first point
     * @param x2 is the second point
     * @return the Euclidean distance
     */
    public double getDistance(String[] x1, String[] x2)
    {
        double distance = 0;
        for (int i = 0; i < x1.length; i++)
        {
            distance += Math.pow((Double.parseDouble(x1[i]) - Double.parseDouble(x2[i])), 2);
        }
        
        return Math.sqrt(distance);
    }
}
